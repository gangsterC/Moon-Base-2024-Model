# TagPro Moon Base 2024 MLP Evaluator Overlay

This project lets you run a trained neural network evaluator on downloaded **TagPro Moon Base 2024 Capture the Flag replay files** and view the predictions live over the TagPro replay using a Tampermonkey browser overlay.

The intended workflow is:

```text
Downloaded .ndjson replay
        ↓
Python precompute script
        ↓
Prediction JSON/CSV
        ↓
Local prediction server
        ↓
Tampermonkey overlay on TagPro replay page
```

This project is currently built for **Moon Base 2024, Capture the Flag** only. The model should not be trusted on other maps or Neutral Flag replays unless a new model is trained for those maps/modes.

---

## What this project does

This project has three main pieces:

1. **Replay precompute script**  
   Converts a downloaded TagPro `.ndjson` replay into model predictions.

2. **Local prediction server**  
   Serves the precomputed prediction JSON file to the browser.

3. **Tampermonkey overlay**  
   Shows xCap, escape, flag-loss, and xScore predictions over the TagPro replay.

The overlay is designed to auto-sync to the replay clock using TagPro's `tagpro.gameEndsAt` clock value.

---

## What the model predicts

The trained MLP outputs calibrated probabilities for these targets:

| Target | Meaning |
|---|---|
| `red_cap_10s` | Probability Red scores within the next 10 seconds |
| `blue_cap_10s` | Probability Blue scores within the next 10 seconds |
| `red_cap_20s` | Probability Red scores within the next 20 seconds |
| `blue_cap_20s` | Probability Blue scores within the next 20 seconds |
| `red_out_base_5s` | Probability Red flag carrier/state is out of base within 5 seconds |
| `blue_out_base_5s` | Probability Blue flag carrier/state is out of base within 5 seconds |
| `red_escape_5s` | Probability Red escapes base within 5 seconds from eligible states |
| `blue_escape_5s` | Probability Blue escapes base within 5 seconds from eligible states |
| `red_fc_lost_2s` | Probability Red flag carrier loses the flag within 2 seconds |
| `blue_fc_lost_2s` | Probability Blue flag carrier loses the flag within 2 seconds |

It also outputs:

| Value | Meaning |
|---|---|
| `red_score_delta_20s` | Expected Red score swing over the next 20 seconds |
| `red_xscore_20s` | Current Red score differential plus expected Red score swing over the next 20 seconds |

### xScore interpretation

`red_xscore_20s` is Red's projected score differential.

Examples:

```text
red_xscore_20s = +1.50 → Red is projected to be ahead by 1.5 caps
red_xscore_20s =  0.00 → roughly even
red_xscore_20s = -1.50 → Blue is projected to be ahead by 1.5 caps
```

The overlay displays xScore as a horizontal evaluator bar, similar in spirit to a chess engine evaluation bar.

---

## Required model files

Place these files in the `model/` folder:

```text
model/
├── best_model.pt
└── calibrators_platt.joblib
```

| File | Purpose |
|---|---|
| `best_model.pt` | The trained PyTorch MLP checkpoint |
| `calibrators_platt.joblib` | Platt-scaling calibrators that turn raw MLP logits into better calibrated probabilities |

The calibrator file is strongly recommended. Without it, the model may rank states well but output poorly calibrated probabilities.

---

## Recommended repository structure

```text
tagpro-moonbase-mlp-evaluator/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── model/
│   ├── best_model.pt
│   ├── calibrators_platt.joblib
│   ├── neural_final_classification_results.csv
│   ├── neural_final_regression_results.csv
│   └── calibration_results_raw_vs_platt.csv
│
├── scripts/
│   ├── moonbase_ctf_xgb_refined_stream_cached.py
│   ├── precompute_mlp_replay_predictions_fixed_raw.py
│   └── tagpro_precomputed_overlay_server.py
│
├── tampermonkey/
│   └── tagpro_mlp_overlay_auto_sync.user.js
│
├── precomputed_predictions/
│   └── .gitkeep
│
└── examples/
    ├── sample_predictions.json
    └── sample_predictions.csv
```

### Important scripts

| File | Purpose |
|---|---|
| `scripts/moonbase_ctf_xgb_refined_stream_cached.py` | Base replay parser / feature builder |
| `scripts/precompute_mlp_replay_predictions_fixed_raw.py` | Converts a downloaded `.ndjson` replay into model predictions |
| `scripts/tagpro_precomputed_overlay_server.py` | Serves prediction JSON files to the browser overlay |
| `tampermonkey/tagpro_mlp_overlay_auto_sync.user.js` | Browser overlay userscript |

---

## System requirements

This project should work on:

- Linux
- macOS
- Windows
- Chromebook Linux / Crostini

You need:

- Python 3.10 or newer recommended
- A browser that supports Tampermonkey
- Downloaded TagPro `.ndjson` replay files
- The trained model files

The model runs on CPU. No GPU is required.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/tagpro-moonbase-mlp-evaluator.git
cd tagpro-moonbase-mlp-evaluator
```

### 2. Create a Python virtual environment

#### Linux / macOS / Chromebook Linux

```bash
python3 -m venv tagpro_mlp_env
source tagpro_mlp_env/bin/activate
```

#### Windows PowerShell

```powershell
python -m venv tagpro_mlp_env
.\tagpro_mlp_env\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again.

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch installation is large or fails on a low-storage system, install the CPU wheel directly:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

On Chromebook/Crostini with limited storage, you may need to clear old datasets or install with no cache:

```bash
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir -r requirements.txt
```

---

## Downloading a TagPro replay

This project expects a downloaded `.ndjson` replay file.

Example file name:

```text
tagpro-bwbqgwlp-aivobyzi.ndjson
```

Put your replay somewhere easy to reference, such as:

```text
replays/
```

Example:

```bash
mkdir -p replays
```

Then place your `.ndjson` file inside:

```text
replays/tagpro-bwbqgwlp-aivobyzi.ndjson
```

---

## Precomputing predictions for a replay

Before using the browser overlay, run the model on the replay and save predictions.

### Linux / macOS / Chromebook Linux

```bash
python scripts/precompute_mlp_replay_predictions_fixed_raw.py \
  --replay "replays/tagpro-bwbqgwlp-aivobyzi.ndjson" \
  --base-script "scripts/moonbase_ctf_xgb_refined_stream_cached.py" \
  --model-dir "model" \
  --calibrator "model/calibrators_platt.joblib" \
  --out-dir "precomputed_predictions" \
  --sample-hz 5 \
  --coord-scale 0.4 \
  --red-escape-progress 0.28 \
  --blue-escape-progress 0.28
```

### Windows PowerShell

```powershell
python scripts/precompute_mlp_replay_predictions_fixed_raw.py `
  --replay "replays/tagpro-bwbqgwlp-aivobyzi.ndjson" `
  --base-script "scripts/moonbase_ctf_xgb_refined_stream_cached.py" `
  --model-dir "model" `
  --calibrator "model/calibrators_platt.joblib" `
  --out-dir "precomputed_predictions" `
  --sample-hz 5 `
  --coord-scale 0.4 `
  --red-escape-progress 0.28 `
  --blue-escape-progress 0.28
```

The script will create files like:

```text
precomputed_predictions/tagpro-bwbqgwlp-aivobyzi_mlp_predictions.json
precomputed_predictions/tagpro-bwbqgwlp-aivobyzi_mlp_predictions.csv
```

The JSON file is used by the overlay. The CSV file is useful for debugging or analysis.

---

## Reading the sanity report

When you precompute predictions, the script prints diagnostic sections.

You should see something like:

```text
=== Feature sanity after full geometry repair ===
Rows parsed: 2,591
t range: 0.60 to 518.60
has_red_fc ...
has_blue_fc ...
red_fc_out ...
blue_fc_out ...

=== Standardized feature sanity ===
abs z p50=...
abs z p95=...
abs z p99=...

=== Prediction sanity after full geometry repair ===
red_cap_10s ...
blue_cap_10s ...
red_cap_20s ...
blue_cap_20s ...
red_xscore_20s ...
```

### Healthy prediction ranges

Values will vary by replay, but generally:

```text
xCap10 usually stays in the low percentages, with spikes during dangerous moments.
xCap20 is usually higher than xCap10.
xScore should not normally swing wildly from -10 to +10 on normal game states.
out_base and escape should not be stuck at 100% or 0% for the entire replay.
```

If everything is stuck at extreme values, the replay is probably not Moon Base 2024, the wrong parser is being used, or the feature pipeline is mismatched.

---

## Running the local prediction server

The server lets the Tampermonkey overlay load your precomputed prediction JSON.

### Linux / macOS / Chromebook Linux

```bash
python scripts/tagpro_precomputed_overlay_server.py \
  --pred-dir "precomputed_predictions" \
  --host 0.0.0.0 \
  --port 8767
```

### Windows PowerShell

```powershell
python scripts/tagpro_precomputed_overlay_server.py `
  --pred-dir "precomputed_predictions" `
  --host 0.0.0.0 `
  --port 8767
```

Leave this terminal open while watching the replay.

---

## Testing the server

Open this in your browser.

### Chromebook Linux / Crostini

```text
http://penguin.linux.test:8767/health
```

### Windows / macOS / normal Linux

```text
http://localhost:8767/health
```

Expected response:

```json
{"ok": true, "service": "tagpro_precomputed_overlay_server"}
```

Then test:

### Chromebook Linux / Crostini

```text
http://penguin.linux.test:8767/list
```

### Windows / macOS / normal Linux

```text
http://localhost:8767/list
```

You should see your prediction JSON files.

---

## Installing the Tampermonkey overlay

1. Install the Tampermonkey browser extension.
2. Open the Tampermonkey Dashboard.
3. Create a new script.
4. Delete the default contents.
5. Paste the contents of:

```text
tampermonkey/tagpro_mlp_overlay_auto_sync.user.js
```

6. Save the script.
7. Make sure the script is enabled.

### Important: server URL setting

Inside the Tampermonkey script, find:

```javascript
const SERVER = "http://penguin.linux.test:8767";
```

Use this for Chromebook Linux / Crostini:

```javascript
const SERVER = "http://penguin.linux.test:8767";
```

Use this for Windows / macOS / normal Linux:

```javascript
const SERVER = "http://localhost:8767";
```

---

## Browser permissions

If the overlay does not appear, check browser/Tampermonkey permissions.

### Chrome / Chromium

Go to:

```text
chrome://extensions
```

Then:

1. Find Tampermonkey.
2. Click **Details**.
3. Enable **Allow User Scripts** if available.
4. Enable **Developer mode** if needed.
5. Set **Site access** to **On all sites** or allow:

```text
https://tagpro.koalabeast.com/*
```

### Quick injection test

If nothing appears, create a tiny Tampermonkey script:

```javascript
// ==UserScript==
// @name         TagPro Injection Test
// @namespace    tagpro-test
// @version      0.1
// @match        https://tagpro.koalabeast.com/game*
// @run-at       document-idle
// @grant        none
// ==/UserScript==

(function () {
  console.log("[TPV TEST] Injected:", window.location.href);
  const box = document.createElement("div");
  box.textContent = "TPV TEST INJECTED";
  box.style.position = "fixed";
  box.style.top = "80px";
  box.style.right = "20px";
  box.style.zIndex = "999999999";
  box.style.background = "black";
  box.style.color = "white";
  box.style.border = "3px solid red";
  box.style.padding = "20px";
  document.body.appendChild(box);
})();
```

If this test does not appear, Tampermonkey is not injecting.

---

## Using the overlay

1. Start the prediction server.
2. Open a TagPro replay page, for example:

```text
https://tagpro.koalabeast.com/game?replay=...
```

3. The overlay should appear in the top-right.
4. Click **Refresh files**.
5. Select the matching precomputed prediction JSON.
6. Click **Load**.
7. Start the TagPro replay.

The overlay uses TagPro's own clock through `tagpro.gameEndsAt`, so it should auto-sync while the replay plays.

The bottom-right of the overlay should show something like:

```text
346/2591 t=69.6 auto:clock
```

If it says `manual`, then auto-sync is not finding TagPro's clock.

---

## Overlay controls

| Control | Purpose |
|---|---|
| `Refresh files` | Reloads the list of prediction JSON files from the local server |
| `Load` | Loads the selected prediction JSON |
| `Auto Sync ON/OFF` | Toggles automatic sync to TagPro's replay clock |
| `Sync 0` | Resets manual timer and offset |
| `▶ Manual` | Manual fallback timer if auto-sync fails |
| `-1.0`, `-0.5`, `+0.5`, `+1.0` | Timing offset adjustments if the overlay appears early/late |

---

## Recommended workflow

For each new replay:

```bash
# 1. Put replay in replays/
# 2. Precompute predictions
python scripts/precompute_mlp_replay_predictions_fixed_raw.py \
  --replay "replays/YOUR_REPLAY.ndjson" \
  --base-script "scripts/moonbase_ctf_xgb_refined_stream_cached.py" \
  --model-dir "model" \
  --calibrator "model/calibrators_platt.joblib" \
  --out-dir "precomputed_predictions" \
  --sample-hz 5 \
  --coord-scale 0.4 \
  --red-escape-progress 0.28 \
  --blue-escape-progress 0.28

# 3. Start server
python scripts/tagpro_precomputed_overlay_server.py \
  --pred-dir "precomputed_predictions" \
  --host 0.0.0.0 \
  --port 8767

# 4. Open TagPro replay
# 5. Load matching prediction JSON in overlay
```

---

## Troubleshooting

### The overlay does not appear

Check:

- Tampermonkey is installed.
- Tampermonkey script is enabled.
- Browser gave Tampermonkey site access.
- The script has this match rule:

```javascript
// @match https://tagpro.koalabeast.com/game*
```

- DevTools console shows:

```text
[TPV] Auto-sync precomputed MLP overlay started
```

### The overlay appears but shows server error

Check that the server is running:

```text
http://localhost:8767/health
```

or on Chromebook:

```text
http://penguin.linux.test:8767/health
```

If the page does not load, start the server again.

### No prediction files show up

Check the prediction folder:

```bash
ls -lh precomputed_predictions
```

Make sure it contains:

```text
*_mlp_predictions.json
```

Then refresh the overlay file list.

### Predictions are extreme or nonsensical

This usually means one of the following:

- The replay is not Moon Base 2024.
- The replay is not CTF.
- The replay file does not match the replay being watched.
- The parser did not reconstruct features correctly.
- A stale old prediction JSON is being loaded.

Fixes:

1. Re-run `precompute_mlp_replay_predictions_fixed_raw.py`.
2. Make sure the terminal sanity report looks reasonable.
3. Restart the server.
4. Refresh files in the overlay.
5. Load the newly generated JSON.

### Auto-sync is wrong

The overlay uses:

```javascript
tagpro.gameEndsAt
```

to compute elapsed time:

```text
elapsed = 480 - seconds_remaining
```

If the overlay is slightly early or late, use:

```text
-0.5 / +0.5
```

If auto-sync fails entirely, toggle **Auto Sync OFF** and use the manual timer.

### PyTorch install fails because of storage

On low-storage systems, try:

```bash
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir -r requirements.txt
```

On Chromebook/Crostini, make sure you have enough Linux storage available.

### Windows path examples

Use quotes around paths:

```powershell
python scripts/precompute_mlp_replay_predictions_fixed_raw.py `
  --replay "C:\Users\YourName\Downloads\tagpro-replay.ndjson" `
  --base-script "scripts\moonbase_ctf_xgb_refined_stream_cached.py" `
  --model-dir "model" `
  --calibrator "model\calibrators_platt.joblib" `
  --out-dir "precomputed_predictions" `
  --sample-hz 5
```

---

## Model performance notes

On the held-out full test set, the MLP achieved strong ranking performance for several key targets after calibration.

Examples:

| Target | AUC |
|---|---:|
| `red_out_base_5s` | ~0.952 |
| `blue_out_base_5s` | ~0.952 |
| `red_escape_5s` | ~0.880 |
| `blue_escape_5s` | ~0.884 |
| `red_cap_10s` | ~0.787 |
| `blue_cap_10s` | ~0.790 |
| `red_fc_lost_2s` | ~0.754 |
| `blue_fc_lost_2s` | ~0.764 |

Calibration significantly improved logloss and Brier score while leaving AUC unchanged.

---

## Limitations

This project is experimental.

Current limitations:

- Only trained for **Moon Base 2024 CTF**.
- Not validated for other maps.
- Not validated for Neutral Flag.
- xScore is useful as a state evaluator but should not be treated as perfect.
- The model is only as good as the replay parser / feature reconstruction.
- Predictions may be unreliable if TagPro replay formats change.

---

## What not to include in GitHub

Do not upload huge local training artifacts unless you intentionally want to publish the dataset.

Avoid committing:

```text
raw replay folders
training shards
validation/test shards
xgb_cache
local_replay_cache
large old experiment folders
```

The public inference repo only needs:

```text
model files
scripts
Tampermonkey overlay
README
requirements
optional examples
```

---

## Suggested `.gitignore`

```gitignore
__pycache__/
*.pyc
.env
.venv/
venv/
tagpro_mlp_env/

replays/
precomputed_predictions/*
!precomputed_predictions/.gitkeep

shards/
xgb_cache/
local_replay_cache/
raw/
*.csv.gz

.DS_Store
Thumbs.db
```

Do not ignore `model/best_model.pt` if you intend to publish the trained model.

---

## License and credit

Choose a license before publishing publicly. If you are not sure, MIT is a simple permissive option.

Also consider crediting:

- TagPro / KoalaBeast for the game/replay ecosystem
- Any contributors to replay parsing or map tooling
- Anyone who helped collect replays or test the overlay
