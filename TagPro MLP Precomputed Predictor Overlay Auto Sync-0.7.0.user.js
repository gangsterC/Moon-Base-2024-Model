// ==UserScript==
// @name         TagPro MLP Precomputed Predictor Overlay Auto Sync
// @namespace    tagpro-mlp-overlay
// @version      0.7.0
// @description  Displays precomputed calibrated MLP xCap/xER/xPop/xScore predictions during TagPro replays, synced to raw replay time. Highlights panel: 30s influential window, game-end aware.
// @match        https://tagpro.koalabeast.com/game*
// @run-at       document-idle
// @grant        GM_xmlhttpRequest
// @grant        unsafeWindow
// @connect      penguin.linux.test
// @connect      127.0.0.1
// @connect      localhost
// ==/UserScript==

(function () {
  "use strict";

  console.log("[TPV] Auto-sync precomputed MLP overlay started on:", window.location.href);

  const SERVER = "http://penguin.linux.test:8767";

  // Length of TagPro gameplay (state == 1) in seconds. Used only by the
  // gameEndsAt fallback path. Predictions themselves are indexed in RAW
  // REPLAY SECONDS (from the first byte of the ndjson recording), which
  // includes the ~20s warmup.
  const GAME_LENGTH_SECONDS = 480;

  let predictions = [];
  let loadedFile = "";
  let loaded = false;

  let activeIndex = 0;

  // Whole-game highlights, computed once after a file loads.
  let highlights = null;

  // Auto-sync state.
  let autoSync = true;
  let offset = 0;
  let lastSyncMode = "auto";
  let lastSyncRaw = "";

  // Manual fallback state.
  let manualTime = 0;
  let manualPlaying = false;
  let lastTick = performance.now();

  let animationStarted = false;

  // ---- WARMUP-OFFSET DETECTION (only used by the gameEndsAt fallback) ----
  //
  // Predictions are indexed by raw replay time, which begins at t=0 when the
  // ndjson recording starts (during warmup). The TagPro `gameEndsAt` value, by
  // contrast, only tracks the in-game clock. So if we have to fall back to
  // gameEndsAt, we must add the wall-clock duration of the warmup phase to map
  // in-game elapsed onto raw replay time.
  //
  // We detect that offset on the fly: when `gameEndsAt - Date.now()` jumps
  // forward by >60s (warmup → state 1 transition), the time elapsed since the
  // page first observed `gameEndsAt` is approximately the warmup duration.
  //
  let pageFirstSeenGameEndsAtMs = null;
  let lastObservedRemainingMs = null;
  let warmupOffsetSec = null;

  function pageWindow() {
    try {
      return unsafeWindow || window;
    } catch (e) {
      return window;
    }
  }

  function reqJson(url) {
    return new Promise((resolve, reject) => {
      GM_xmlhttpRequest({
        method: "GET",
        url,
        timeout: 30000,
        onload: (res) => {
          try {
            resolve(JSON.parse(res.responseText));
          } catch (e) {
            reject(new Error("Could not parse JSON: " + e.message));
          }
        },
        ontimeout: () => reject(new Error("Request timed out")),
        onerror: () => reject(new Error("Request failed")),
      });
    });
  }

  function pct(x) {
    if (x === undefined || x === null || Number.isNaN(x)) return "--";
    const v = x * 100;

    if (v > 0 && v < 0.1) return "<0.1%";
    if (v > 99.9 && v < 100) return ">99.9%";

    return `${v.toFixed(1)}%`;
  }

  function num(x, digits = 2) {
    if (x === undefined || x === null || Number.isNaN(x)) return "--";
    return Number(x).toFixed(digits);
  }

  function dangerClass(p) {
    if (p === undefined || p === null || Number.isNaN(p)) return "tpv-low";
    if (p >= 0.50) return "tpv-hot";
    if (p >= 0.25) return "tpv-warn";
    if (p >= 0.10) return "tpv-mid";
    return "tpv-low";
  }

  function injectStyle() {
    const style = document.createElement("style");
    style.textContent = `
      #tpv-panel {
        position: fixed;
        top: 72px;
        right: 16px;
        z-index: 999999999;
        width: 292px;
        font-family: Arial, Helvetica, sans-serif;
        font-size: 12px;
        color: #fff;
        background: rgba(12, 14, 18, 0.82);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 10px;
        box-shadow: 0 8px 22px rgba(0,0,0,0.35);
        overflow: hidden;
        backdrop-filter: blur(3px);
        user-select: none;
      }

      #tpv-header {
        cursor: move;
        padding: 8px 10px;
        font-weight: 700;
        letter-spacing: .2px;
        background: rgba(255,255,255,0.10);
        display: flex;
        justify-content: space-between;
        align-items: center;
      }

      #tpv-body {
        padding: 8px 10px 10px 10px;
      }

      #tpv-status {
        font-size: 11px;
        color: #ddd;
        margin-bottom: 6px;
        line-height: 1.25;
      }

      #tpv-select {
        width: 100%;
        margin: 5px 0;
        font-size: 11px;
        background: rgba(255,255,255,0.92);
        color: black;
      }

      .tpv-btn {
        font-size: 11px;
        padding: 3px 6px;
        margin: 2px;
        border: 1px solid rgba(255,255,255,.22);
        border-radius: 5px;
        background: rgba(255,255,255,.10);
        color: white;
        cursor: pointer;
      }

      .tpv-btn:hover {
        background: rgba(255,255,255,.20);
      }

      .tpv-btn-active {
        background: rgba(88,166,255,.35) !important;
        border-color: rgba(88,166,255,.75) !important;
      }

      .tpv-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 7px;
        margin-top: 7px;
      }

      .tpv-card {
        border-radius: 8px;
        padding: 7px;
        background: rgba(255,255,255,0.08);
      }

      .tpv-red {
        border-left: 3px solid #ff5a5a;
      }

      .tpv-blue {
        border-left: 3px solid #58a6ff;
      }

      .tpv-team {
        font-size: 12px;
        font-weight: 800;
        margin-bottom: 5px;
      }

      .tpv-main {
        font-size: 22px;
        font-weight: 900;
        line-height: 1;
        margin-bottom: 5px;
      }

      .tpv-row {
        display: flex;
        justify-content: space-between;
        gap: 8px;
        line-height: 1.5;
        border-top: 1px solid rgba(255,255,255,0.07);
        padding-top: 2px;
      }

      .tpv-label {
        color: rgba(255,255,255,.72);
      }

      .tpv-low {
        color: #d6d6d6;
      }

      .tpv-mid {
        color: #65d77b;
      }

      .tpv-warn {
        color: #ffd166;
      }

      .tpv-hot {
        color: #ff6b6b;
      }

      #tpv-xscore-wrap {
        margin-top: 9px;
        padding: 7px;
        border-radius: 8px;
        background: rgba(255,255,255,0.08);
      }

      #tpv-xscore-labels {
        display: flex;
        justify-content: space-between;
        font-size: 11px;
        margin-bottom: 4px;
      }

      #tpv-xscore-blue-label {
        color: #58a6ff;
        font-weight: 700;
      }

      #tpv-xscore-red-label {
        color: #ff5a5a;
        font-weight: 700;
      }

      #tpv-xscore-bar {
        position: relative;
        height: 14px;
        border-radius: 999px;
        overflow: hidden;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.18);
      }

      #tpv-xscore-blue-fill {
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        background: rgba(88,166,255,0.82);
        width: 50%;
        transition: width 0.10s linear;
      }

      #tpv-xscore-red-fill {
        position: absolute;
        right: 0;
        top: 0;
        bottom: 0;
        background: rgba(255,90,90,0.82);
        width: 50%;
        transition: width 0.10s linear;
      }

      #tpv-xscore-zero {
        position: absolute;
        top: -2px;
        bottom: -2px;
        left: 50%;
        width: 2px;
        background: rgba(255,255,255,0.85);
        transform: translateX(-1px);
        z-index: 3;
      }

      .tpv-footer {
        margin-top: 7px;
        font-size: 11px;
        color: #ccc;
        display: flex;
        justify-content: space-between;
        gap: 8px;
      }

      #tpv-file {
        margin-top: 5px;
        font-size: 10px;
        color: rgba(255,255,255,0.65);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      #tpv-sync-info {
        margin-top: 4px;
        font-size: 10px;
        color: rgba(255,255,255,0.55);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      #tpv-highlights {
        margin-top: 9px;
        padding: 8px 9px 9px 9px;
        border-radius: 8px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
      }

      #tpv-hl-title {
        font-size: 11px;
        font-weight: 800;
        letter-spacing: .5px;
        color: rgba(255,255,255,0.85);
        margin-bottom: 5px;
        text-transform: uppercase;
      }

      .tpv-hl-row {
        margin-top: 5px;
      }

      .tpv-hl-row:first-child {
        margin-top: 0;
      }

      .tpv-hl-label {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: .35px;
        color: rgba(255,255,255,0.55);
        margin-bottom: 1px;
      }

      .tpv-hl-vals {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 6px;
        font-size: 11px;
        font-variant-numeric: tabular-nums;
      }

      .tpv-hl-red {
        color: #ff8585;
      }

      .tpv-hl-blue {
        color: #80b5ff;
      }

      .tpv-hl-na {
        color: rgba(255,255,255,0.35);
      }
    `;
    document.head.appendChild(style);
  }

  function createPanel() {
    const panel = document.createElement("div");
    panel.id = "tpv-panel";
    panel.innerHTML = `
      <div id="tpv-header">
        <span>MLP Predictor</span>
        <span id="tpv-clock">0.0s</span>
      </div>

      <div id="tpv-body">
        <div id="tpv-status">Connecting...</div>

        <select id="tpv-select"></select>

        <div>
          <button class="tpv-btn" id="tpv-refresh">Refresh files</button>
          <button class="tpv-btn" id="tpv-load">Load</button>
        </div>

        <div>
          <button class="tpv-btn tpv-btn-active" id="tpv-auto">Auto Sync ON</button>
          <button class="tpv-btn" id="tpv-sync0">Sync 0</button>
          <button class="tpv-btn" id="tpv-play">▶ Manual</button>
        </div>

        <div>
          <button class="tpv-btn" id="tpv-minus1">-1.0</button>
          <button class="tpv-btn" id="tpv-minus">-0.5</button>
          <button class="tpv-btn" id="tpv-plus">+0.5</button>
          <button class="tpv-btn" id="tpv-plus1">+1.0</button>
        </div>

        <div class="tpv-grid">
          <div class="tpv-card tpv-red">
            <div class="tpv-team">RED</div>
            <div class="tpv-main" id="tpv-red-main">--</div>
            <div class="tpv-row"><span class="tpv-label">Cap20</span><span id="tpv-red-cap20">--</span></div>
            <div class="tpv-row"><span class="tpv-label">Out5</span><span id="tpv-red-out">--</span></div>
            <div class="tpv-row"><span class="tpv-label">Esc5</span><span id="tpv-red-esc">--</span></div>
            <div class="tpv-row"><span class="tpv-label">Lost2</span><span id="tpv-red-lost">--</span></div>
          </div>

          <div class="tpv-card tpv-blue">
            <div class="tpv-team">BLUE</div>
            <div class="tpv-main" id="tpv-blue-main">--</div>
            <div class="tpv-row"><span class="tpv-label">Cap20</span><span id="tpv-blue-cap20">--</span></div>
            <div class="tpv-row"><span class="tpv-label">Out5</span><span id="tpv-blue-out">--</span></div>
            <div class="tpv-row"><span class="tpv-label">Esc5</span><span id="tpv-blue-esc">--</span></div>
            <div class="tpv-row"><span class="tpv-label">Lost2</span><span id="tpv-blue-lost">--</span></div>
          </div>
        </div>

        <div id="tpv-xscore-wrap">
          <div id="tpv-xscore-labels">
            <span id="tpv-xscore-blue-label">Blue --</span>
            <span id="tpv-xscore-red-label">Red --</span>
          </div>
          <div id="tpv-xscore-bar">
            <div id="tpv-xscore-blue-fill"></div>
            <div id="tpv-xscore-red-fill"></div>
            <div id="tpv-xscore-zero"></div>
          </div>
        </div>

        <div class="tpv-footer">
          <span id="tpv-xscore">xScore20: --</span>
          <span id="tpv-meta">-- rows</span>
        </div>

        <div id="tpv-file">No file loaded</div>
        <div id="tpv-sync-info">sync: --</div>

        <div id="tpv-highlights" style="display:none;">
          <div id="tpv-hl-title">Highlights</div>
          <div id="tpv-hl-content"></div>
        </div>
      </div>
    `;

    document.body.appendChild(panel);

    document.getElementById("tpv-refresh").onclick = refreshFiles;
    document.getElementById("tpv-load").onclick = loadSelected;

    document.getElementById("tpv-auto").onclick = () => {
      autoSync = !autoSync;
      updateAutoButton();
      lastTick = performance.now();
      updateOnce();
    };

    document.getElementById("tpv-sync0").onclick = () => {
      manualTime = 0;
      offset = 0;
      activeIndex = 0;
      // Also reset warmup-offset detection so a fresh probe can run.
      pageFirstSeenGameEndsAtMs = null;
      lastObservedRemainingMs = null;
      warmupOffsetSec = null;
      lastTick = performance.now();
      updateOnce();
    };

    document.getElementById("tpv-play").onclick = () => {
      manualPlaying = !manualPlaying;
      lastTick = performance.now();
      document.getElementById("tpv-play").textContent = manualPlaying ? "⏸ Manual" : "▶ Manual";
    };

    document.getElementById("tpv-minus1").onclick = () => {
      offset -= 1.0;
      updateOnce();
    };

    document.getElementById("tpv-minus").onclick = () => {
      offset -= 0.5;
      updateOnce();
    };

    document.getElementById("tpv-plus").onclick = () => {
      offset += 0.5;
      updateOnce();
    };

    document.getElementById("tpv-plus1").onclick = () => {
      offset += 1.0;
      updateOnce();
    };

    makeDraggable(panel, document.getElementById("tpv-header"));
    updateAutoButton();
  }

  function updateAutoButton() {
    const btn = document.getElementById("tpv-auto");
    if (!btn) return;

    if (autoSync) {
      btn.textContent = "Auto Sync ON";
      btn.classList.add("tpv-btn-active");
    } else {
      btn.textContent = "Auto Sync OFF";
      btn.classList.remove("tpv-btn-active");
    }
  }

  function makeDraggable(panel, handle) {
    let dragging = false;
    let startX = 0;
    let startY = 0;
    let startLeft = 0;
    let startTop = 0;

    handle.addEventListener("mousedown", (e) => {
      dragging = true;
      startX = e.clientX;
      startY = e.clientY;

      const r = panel.getBoundingClientRect();
      startLeft = r.left;
      startTop = r.top;

      panel.style.left = `${startLeft}px`;
      panel.style.top = `${startTop}px`;
      panel.style.right = "auto";

      e.preventDefault();
    });

    window.addEventListener("mousemove", (e) => {
      if (!dragging) return;
      panel.style.left = `${startLeft + e.clientX - startX}px`;
      panel.style.top = `${startTop + e.clientY - startY}px`;
    });

    window.addEventListener("mouseup", () => {
      dragging = false;
    });
  }

  async function refreshFiles() {
    const status = document.getElementById("tpv-status");
    const select = document.getElementById("tpv-select");

    status.textContent = "Loading precomputed prediction list...";

    try {
      const data = await reqJson(`${SERVER}/list`);
      select.innerHTML = "";

      for (const f of data.files || []) {
        const opt = document.createElement("option");
        opt.value = f.name;
        opt.textContent = f.name;
        select.appendChild(opt);
      }

      status.textContent = `Found ${(data.files || []).length} prediction files. Pick one, then Load.`;
    } catch (e) {
      status.textContent = `Server/list error: ${e.message || e}`;
    }
  }

  async function loadSelected() {
    const status = document.getElementById("tpv-status");
    const select = document.getElementById("tpv-select");
    const file = select.value;

    if (!file) {
      status.textContent = "No prediction file selected.";
      return;
    }

    status.textContent = `Loading ${file}...`;

    // Clear stale highlights immediately; a failed load shouldn't show old data.
    highlights = null;
    renderHighlights(null);

    try {
      const data = await reqJson(`${SERVER}/predict?file=${encodeURIComponent(file)}`);

      if (!data.ok) {
        throw new Error(data.error || "unknown server error");
      }

      predictions = data.predictions || [];
      loadedFile = data.file || file;
      loaded = predictions.length > 0;

      activeIndex = 0;
      manualTime = 0;
      offset = 0;
      manualPlaying = false;
      // Reset warmup detection too: a freshly loaded file may correspond to a
      // different replay or a different load order.
      pageFirstSeenGameEndsAtMs = null;
      lastObservedRemainingMs = null;
      warmupOffsetSec = null;
      lastTick = performance.now();

      // Whole-game highlights are static once the file is loaded.
      highlights = computeHighlights(predictions);
      renderHighlights(highlights);

      document.getElementById("tpv-play").textContent = "▶ Manual";
      document.getElementById("tpv-meta").textContent = `${predictions.length} rows`;
      document.getElementById("tpv-file").textContent = loadedFile;
      status.textContent = `${loadedFile} loaded (${data.calibrated ? "calibrated" : "raw"}, ${predictions.length} rows).`;

      updateOnce();
    } catch (e) {
      status.textContent = `Predict/load error: ${e.message || e}`;
    }
  }

  // ---------------------------------------------------------------------------
  //  TIME RESOLUTION
  // ---------------------------------------------------------------------------
  //
  // Predictions are indexed by RAW REPLAY TIME in seconds, measured from the
  // first event in the ndjson recording. That includes the ~20s warmup plus
  // the post-game tail, so a typical replay's predictions span e.g. 0.6 → 518.6.
  //
  // The previous version used `tagpro.gameEndsAt`, which only tracks the
  // in-game clock (0 → 480). That made the lookup ~20s out of phase during
  // gameplay AND made the panel rocket through end-of-game predictions during
  // the warmup period (when computed in-game elapsed = 460-479). That is the
  // "extreme and frozen" behavior.
  //
  // Below we try several time sources, in priority order:
  //   1. replayIO.{time, currentTime, replayTime, t, elapsed}      (raw replay time directly)
  //   2. replayIO.{tick, frame} / 60                                (ditto via tick count)
  //   3. tagpro.replay.{...}  same fields                           (alt location)
  //   4. tagpro.gameEndsAt + auto-detected warmup offset            (last resort)
  //
  function tryDirectTimeKey(obj, k) {
    const v = obj[k];
    if (!Number.isFinite(v)) return null;
    let seconds = v;
    if (Math.abs(seconds) > 10000) seconds = seconds / 1000; // ms → s
    if (seconds < -5 || seconds > 7200) return null; // sanity: 0–2h
    return seconds;
  }

  function getRawReplayTime() {
    if (!autoSync) return null;

    const w = pageWindow();
    const tp = w.tagpro;
    const rio = w.replayIO;

    const directKeys = ["time", "currentTime", "replayTime", "t", "elapsed"];

    // 1. replayIO direct time
    if (rio) {
      for (const k of directKeys) {
        const s = tryDirectTimeKey(rio, k);
        if (s !== null) {
          lastSyncMode = `auto:rio.${k}`;
          lastSyncRaw = `${k}=${rio[k]}`;
          return s + offset;
        }
      }
      if (Number.isFinite(rio.tick)) {
        lastSyncMode = "auto:rio.tick";
        lastSyncRaw = `tick=${rio.tick}`;
        return rio.tick / 60 + offset;
      }
      if (Number.isFinite(rio.frame)) {
        lastSyncMode = "auto:rio.frame";
        lastSyncRaw = `frame=${rio.frame}`;
        return rio.frame / 60 + offset;
      }
    }

    // 2. tagpro.replay direct time (some replay players hang it here)
    if (tp && tp.replay) {
      for (const k of directKeys) {
        const s = tryDirectTimeKey(tp.replay, k);
        if (s !== null) {
          lastSyncMode = `auto:tp.replay.${k}`;
          lastSyncRaw = `${k}=${tp.replay[k]}`;
          return s + offset;
        }
      }
      if (Number.isFinite(tp.replay.tick)) {
        lastSyncMode = "auto:tp.replay.tick";
        lastSyncRaw = `tick=${tp.replay.tick}`;
        return tp.replay.tick / 60 + offset;
      }
      if (Number.isFinite(tp.replay.frame)) {
        lastSyncMode = "auto:tp.replay.frame";
        lastSyncRaw = `frame=${tp.replay.frame}`;
        return tp.replay.frame / 60 + offset;
      }
    }

    // 3. tagpro.gameEndsAt fallback. This only gives in-game elapsed; we have
    //    to add the warmup duration to land on raw replay time. We auto-detect
    //    the warmup duration by watching for the gameEndsAt forward jump that
    //    happens when state changes from 3 (warmup) to 1 (gameplay).
    if (tp && Number.isFinite(tp.gameEndsAt)) {
      const now = Date.now();
      const remainingMs = tp.gameEndsAt - now;

      if (pageFirstSeenGameEndsAtMs === null) {
        pageFirstSeenGameEndsAtMs = now;
      }

      if (lastObservedRemainingMs !== null && warmupOffsetSec === null) {
        const jumpMs = remainingMs - lastObservedRemainingMs;
        // A jump > 60 seconds forward is unmistakably the warmup→gameplay
        // transition (warmup ≈ 20s, gameplay ≈ 480s).
        if (jumpMs > 60_000) {
          warmupOffsetSec = (now - pageFirstSeenGameEndsAtMs) / 1000;
          console.log(
            "[TPV] Detected warmup → gameplay transition. Warmup offset:",
            warmupOffsetSec.toFixed(2), "s"
          );
        }
      }
      lastObservedRemainingMs = remainingMs;

      const inGameElapsed = GAME_LENGTH_SECONDS - remainingMs / 1000;

      // Until we've detected the warmup end, we have to guess. The safest
      // guess is "we're still in warmup", which means we should NOT use the
      // misleading inGameElapsed (which starts near 460s during warmup). In
      // that case, return null and let manual mode take over until the
      // transition fires.
      if (warmupOffsetSec === null) {
        // If inGameElapsed is plausibly small (<60s past 0), the script may
        // have loaded after warmup ended. Use it directly with offset=0.
        if (inGameElapsed >= -2 && inGameElapsed < 60) {
          lastSyncMode = "auto:gameEndsAt(no-warmup)";
          lastSyncRaw = `inGame=${inGameElapsed.toFixed(2)}`;
          return inGameElapsed + offset;
        }
        // Otherwise we don't trust it. Fall through to manual.
        lastSyncMode = "auto:gameEndsAt(WAITING for warmup probe)";
        lastSyncRaw = `inGame=${inGameElapsed.toFixed(2)}`;
        return null;
      }

      lastSyncMode = "auto:gameEndsAt+warmup";
      lastSyncRaw = `inGame=${inGameElapsed.toFixed(2)}, warmup=${warmupOffsetSec.toFixed(2)}`;
      return inGameElapsed + warmupOffsetSec + offset;
    }

    return null;
  }

  // ---------------------------------------------------------------------------
  //  WHOLE-GAME HIGHLIGHTS
  // ---------------------------------------------------------------------------
  //
  //  Computed once on file load; static for the rest of the session. All
  //  timestamps are in raw replay seconds (same axis as predictions[].t and
  //  the panel clock at the top).
  //
  //  Cap events are derived from `current_score_diff` transitions:
  //    diff increases (e.g. 0 → 1)  ->  red cap
  //    diff decreases (e.g. 1 → 0)  ->  blue cap
  //  Verified against the score events in the source ndjson.
  //
  //  Categories produced (each per-team):
  //    1. unlikelyCap   - lowest xCap_10s (scoring team) at t_cap - 5s
  //    2. likelyCap     - highest xCap_10s (scoring team) at t_cap - 5s
  //    3. influential   - largest xScore Δ over a 30s window in this team's favor
  //                       red:  delta = red_xscore_20s(t+30) - red_xscore_20s(t)   maximized
  //                       blue: same delta minimized (most negative); displayed as
  //                       a positive magnitude in blue's POV
  //    4. hiCapNoCap    - highest xCap_10s where no cap by that team in [t, t+15],
  //                       with predictions whose 15s window crosses game-end
  //                       excluded (a flag carrier who didn't score because the
  //                       clock ran out is not a "model thought a cap was coming"
  //                       miss; it's a clock-running-out miss).
  //
  //  Game-end detection: the predictions trace freezes the moment gameplay
  //  ends (every player's position becomes static, so every model input is
  //  identical from then on). We find the start of that frozen tail by
  //  walking backwards from the last frame; on the test replay this lands
  //  within 0.2s of the actual `end` event in the ndjson.
  //
  //  Edge handling: predictions whose t+15s exceeds the recording end are
  //  excluded from category 4 (we can't verify the no-cap-in-15s claim past
  //  the data we have).
  //
  function computeHighlights(preds) {
    if (!preds || preds.length < 2) return null;

    const tFirst = preds[0].t;
    const tLast = preds[preds.length - 1].t;

    // 1. Detect cap events.
    const caps = [];
    for (let i = 1; i < preds.length; i++) {
      const prev = preds[i - 1].current_score_diff;
      const curr = preds[i].current_score_diff;
      if (curr > prev + 0.5) {
        caps.push({ team: "red", t: preds[i].t });
      } else if (curr < prev - 0.5) {
        caps.push({ team: "blue", t: preds[i].t });
      }
    }

    // Helper: binary search for the prediction whose t is the largest one
    // not exceeding the requested time. Returns null if out of range.
    function predAt(t) {
      if (t < tFirst - 0.2 || t > tLast + 0.2) return null;
      let lo = 0;
      let hi = preds.length - 1;
      while (lo < hi) {
        const mid = (lo + hi + 1) >> 1;
        if (preds[mid].t <= t + 1e-9) lo = mid;
        else hi = mid - 1;
      }
      return preds[lo];
    }

    const out = {
      unlikelyCap: { red: null, blue: null },
      likelyCap:   { red: null, blue: null },
      influential: { red: null, blue: null },
      hiCapNoCap:  { red: null, blue: null },
    };

    // 1 & 2.  For each cap, pull the xCap_10s for the scoring team 5s before.
    for (const cap of caps) {
      const pre = predAt(cap.t - 5);
      if (!pre) continue;
      const xcap = cap.team === "red" ? pre.red_cap_10s : pre.blue_cap_10s;
      if (!Number.isFinite(xcap)) continue;

      const entry = { t: cap.t, prob: xcap };

      const cur1 = out.unlikelyCap[cap.team];
      if (!cur1 || xcap < cur1.prob) out.unlikelyCap[cap.team] = entry;

      const cur2 = out.likelyCap[cap.team];
      if (!cur2 || xcap > cur2.prob) out.likelyCap[cap.team] = entry;
    }

    // 3.  Largest 30s xScore Δ in each team's favor.
    for (let i = 0; i < preds.length; i++) {
      const t0 = preds[i].t;
      const xs0 = preds[i].red_xscore_20s;
      if (!Number.isFinite(xs0)) continue;

      const later = predAt(t0 + 30);
      if (!later) continue;
      const xs1 = later.red_xscore_20s;
      if (!Number.isFinite(xs1)) continue;

      const delta = xs1 - xs0;

      // Red benefits from the most positive delta.
      if (!out.influential.red || delta > out.influential.red.prob) {
        out.influential.red = { t: t0, prob: delta };
      }
      // Blue benefits from the most negative delta. We store the raw signed
      // delta; the renderer will display |delta| in blue's POV.
      if (!out.influential.blue || delta < out.influential.blue.prob) {
        out.influential.blue = { t: t0, prob: delta };
      }
    }

    // 4.  Highest xCap_10s where no cap by that team in next 15s.
    //
    // Game-end detection: predictions freeze (all model inputs identical) the
    // moment the game ends, so the start of the post-game frozen tail marks
    // game over. We exclude any prediction whose 15s lookahead window crosses
    // that boundary, so a flag carrier who simply ran out of clock doesn't
    // get scored as a "the model expected a cap that never came" miss.
    function findGameEndTime() {
      if (preds.length < 6) return tLast;

      const keys = [
        "red_cap_10s", "blue_cap_10s", "red_xscore_20s",
        "red_fc_lost_2s", "blue_fc_lost_2s",
        "red_out_base_5s", "blue_out_base_5s",
        "red_escape_5s", "blue_escape_5s",
      ];
      const EPS = 1e-6;
      const last = preds[preds.length - 1];

      let i = preds.length - 1;
      while (i > 0) {
        let same = true;
        for (const k of keys) {
          if (Math.abs(preds[i - 1][k] - last[k]) > EPS) {
            same = false;
            break;
          }
        }
        if (!same) break;
        i--;
      }

      // Require a meaningful frozen tail (>= 1s = 5 frames at 5Hz). If the
      // entire trace is "frozen" or the freeze is too short to be game-over,
      // fall back to the recording end.
      const frozenFrames = preds.length - i;
      if (frozenFrames < 5 || i === 0) return tLast;
      return preds[i].t;
    }

    const gameEndT = findGameEndTime();

    const redCapTimes  = caps.filter((c) => c.team === "red").map((c) => c.t);
    const blueCapTimes = caps.filter((c) => c.team === "blue").map((c) => c.t);

    function nextCapAfter(t, capTimes) {
      let lo = 0;
      let hi = capTimes.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (capTimes[mid] <= t) lo = mid + 1;
        else hi = mid;
      }
      return lo < capTimes.length ? capTimes[lo] : Infinity;
    }

    // Cutoff: t + 15 must fit within game-end. Equivalently, t < gameEndT - 15.
    const tCutoff = gameEndT - 15;
    for (const p of preds) {
      if (p.t > tCutoff) break;

      if (Number.isFinite(p.red_cap_10s)) {
        const nextRed = nextCapAfter(p.t, redCapTimes);
        if (nextRed - p.t > 15) {
          const cur = out.hiCapNoCap.red;
          if (!cur || p.red_cap_10s > cur.prob) {
            out.hiCapNoCap.red = { t: p.t, prob: p.red_cap_10s };
          }
        }
      }

      if (Number.isFinite(p.blue_cap_10s)) {
        const nextBlue = nextCapAfter(p.t, blueCapTimes);
        if (nextBlue - p.t > 15) {
          const cur = out.hiCapNoCap.blue;
          if (!cur || p.blue_cap_10s > cur.prob) {
            out.hiCapNoCap.blue = { t: p.t, prob: p.blue_cap_10s };
          }
        }
      }
    }

    return out;
  }

  // MM:SS formatting for raw replay seconds.
  function formatT(t) {
    if (!Number.isFinite(t)) return "--";
    const total = Math.max(0, Math.floor(t));
    const m = Math.floor(total / 60);
    const s = total % 60;
    return `${m}:${s.toString().padStart(2, "0")}`;
  }

  function renderHighlights(hl) {
    const wrap = document.getElementById("tpv-highlights");
    const content = document.getElementById("tpv-hl-content");
    if (!wrap || !content) return;

    if (!hl) {
      wrap.style.display = "none";
      content.innerHTML = "";
      return;
    }

    function pctStr(p) {
      if (!Number.isFinite(p)) return "--";
      const v = p * 100;
      if (v > 0 && v < 0.1) return "<0.1%";
      if (v > 99.9 && v < 100) return ">99.9%";
      return `${v.toFixed(1)}%`;
    }

    function deltaStr(team, d) {
      // For category 3: blue's most influential play stores a negative delta.
      // Display each team's value as a positive number in their own POV.
      if (!Number.isFinite(d)) return "--";
      const mag = team === "blue" ? -d : d;
      const sign = mag >= 0 ? "+" : "-";
      return `${sign}${Math.abs(mag).toFixed(2)}`;
    }

    function cell(team, entry, fmtVal) {
      const cls = team === "red" ? "tpv-hl-red" : "tpv-hl-blue";
      if (!entry) {
        return `<span class="${cls} tpv-hl-na">--</span>`;
      }
      const tag = team === "red" ? "R" : "B";
      const v = fmtVal(team, entry.prob);
      return `<span class="${cls}" title="raw t=${entry.t.toFixed(1)}s">${tag} ${v} @ ${formatT(entry.t)}</span>`;
    }

    function row(label, redEntry, blueEntry, fmtVal) {
      return `
        <div class="tpv-hl-row">
          <div class="tpv-hl-label">${label}</div>
          <div class="tpv-hl-vals">
            ${cell("red", redEntry, fmtVal)}
            ${cell("blue", blueEntry, fmtVal)}
          </div>
        </div>
      `;
    }

    const fmtPct = (_team, p) => pctStr(p);
    const fmtDelta = (team, d) => deltaStr(team, d);

    content.innerHTML = `
      ${row("Most unlikely cap (5s pre)", hl.unlikelyCap.red, hl.unlikelyCap.blue, fmtPct)}
      ${row("Most predictable cap (5s pre)", hl.likelyCap.red, hl.likelyCap.blue, fmtPct)}
      ${row("Most influential play (30s xScore Δ)", hl.influential.red, hl.influential.blue, fmtDelta)}
      ${row("Highest xCap, no cap in next 15s", hl.hiCapNoCap.red, hl.hiCapNoCap.blue, fmtPct)}
    `;
    wrap.style.display = "block";
  }

  function currentTime() {
    const autoT = getRawReplayTime();

    if (autoT !== null) {
      manualTime = autoT - offset;
      lastTick = performance.now();
      return autoT;
    }

    lastSyncMode = lastSyncMode.startsWith("auto:") ? lastSyncMode : "manual";

    const now = performance.now();

    if (manualPlaying) {
      manualTime += (now - lastTick) / 1000;
    }

    lastTick = now;
    return manualTime + offset;
  }

  function nearestPrediction(t) {
    if (!loaded || predictions.length === 0) return null;

    // Out-of-range handling: don't show garbage if t is before the first
    // prediction or after the last one.
    if (t < predictions[0].t - 0.5) return null;
    if (t > predictions[predictions.length - 1].t + 0.5) return null;

    while (activeIndex < predictions.length - 1 && predictions[activeIndex + 1].t <= t) {
      activeIndex++;
    }

    while (activeIndex > 0 && predictions[activeIndex].t > t) {
      activeIndex--;
    }

    return predictions[activeIndex];
  }

  function setMetric(id, value) {
    const el = document.getElementById(id);
    if (!el) return;

    el.textContent = pct(value);
    el.className = dangerClass(value);
  }

  function clearMetrics() {
    const ids = [
      "tpv-red-main", "tpv-blue-main",
      "tpv-red-cap20", "tpv-blue-cap20",
      "tpv-red-out", "tpv-blue-out",
      "tpv-red-esc", "tpv-blue-esc",
      "tpv-red-lost", "tpv-blue-lost",
    ];
    for (const id of ids) {
      const el = document.getElementById(id);
      if (!el) continue;
      el.textContent = "--";
      el.className = "tpv-low";
    }
    setXScoreBar(null);
  }

  function setXScoreBar(redXScore) {
    const blueLabel = document.getElementById("tpv-xscore-blue-label");
    const redLabel = document.getElementById("tpv-xscore-red-label");
    const blueFill = document.getElementById("tpv-xscore-blue-fill");
    const redFill = document.getElementById("tpv-xscore-red-fill");

    if (!blueLabel || !redLabel || !blueFill || !redFill) return;

    if (redXScore === undefined || redXScore === null || Number.isNaN(redXScore)) {
      blueLabel.textContent = "Blue --";
      redLabel.textContent = "Red --";
      blueFill.style.width = "50%";
      redFill.style.width = "50%";
      return;
    }

    const mercy = 5.0;
    const clipped = Math.max(-mercy, Math.min(mercy, redXScore));
    const redFrac = (clipped + mercy) / (2 * mercy);
    const blueFrac = 1 - redFrac;
    const blueXScore = -redXScore;

    blueLabel.textContent = `Blue ${blueXScore >= 0 ? "+" : ""}${blueXScore.toFixed(2)}`;
    redLabel.textContent = `Red ${redXScore >= 0 ? "+" : ""}${redXScore.toFixed(2)}`;

    blueFill.style.width = `${blueFrac * 100}%`;
    redFill.style.width = `${redFrac * 100}%`;
  }

  function updateOnce() {
    const t = currentTime();

    const clock = document.getElementById("tpv-clock");
    if (clock) clock.textContent = `${t.toFixed(1)}s`;

    const r = nearestPrediction(t);

    const syncInfo = document.getElementById("tpv-sync-info");
    if (syncInfo) {
      syncInfo.textContent = `sync: ${lastSyncMode}${lastSyncRaw ? "  (" + lastSyncRaw + ")" : ""}`;
    }

    if (!r) {
      // Out of range or not loaded. Blank the panel rather than showing
      // misleading frozen values.
      clearMetrics();
      const meta = document.getElementById("tpv-meta");
      if (meta) {
        const off = offset === 0 ? "" : ` off=${offset.toFixed(1)}`;
        const status = !loaded ? "not loaded" : "out of range";
        meta.textContent = `${status} t=${t.toFixed(1)}${off}`;
      }
      const xscore = document.getElementById("tpv-xscore");
      if (xscore) xscore.textContent = `xScore20: --`;
      return;
    }

    setMetric("tpv-red-main", r.red_cap_10s);
    setMetric("tpv-blue-main", r.blue_cap_10s);

    setMetric("tpv-red-cap20", r.red_cap_20s);
    setMetric("tpv-blue-cap20", r.blue_cap_20s);

    setMetric("tpv-red-out", r.red_out_base_5s);
    setMetric("tpv-blue-out", r.blue_out_base_5s);

    setMetric("tpv-red-esc", r.red_escape_5s);
    setMetric("tpv-blue-esc", r.blue_escape_5s);

    setMetric("tpv-red-lost", r.red_fc_lost_2s);
    setMetric("tpv-blue-lost", r.blue_fc_lost_2s);

    const xs = r.red_xscore_20s;
    setXScoreBar(xs);

    const xscore = document.getElementById("tpv-xscore");
    if (xscore) {
      xscore.textContent = `xScore20: ${num(xs, 2)} red POV`;
    }

    const meta = document.getElementById("tpv-meta");
    if (meta) {
      const off = offset === 0 ? "" : ` off=${offset.toFixed(1)}`;
      meta.textContent = `${activeIndex + 1}/${predictions.length} t=${r.t.toFixed(1)}${off}`;
    }
  }

  function animationLoop() {
    updateOnce();
    requestAnimationFrame(animationLoop);
  }

  async function boot() {
    injectStyle();
    createPanel();
    await refreshFiles();

    if (!animationStarted) {
      animationStarted = true;
      animationLoop();
    }
  }

  boot();
})();