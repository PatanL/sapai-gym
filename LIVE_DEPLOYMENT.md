# Live Super Auto Pets deployment

## Canvas contract

The first supported browser/desktop profile is:

- viewport: **1920 × 1080**
- aspect ratio: **16:9**
- device scale factor: **1.0**
- browser chrome: excluded
- Unity canvas: full viewport (`x=0`, `y=0`, `width=1920`, `height=1080`)

This was measured against the official Unity WebGL client. The 2048×1680
background files in `sap-replay-bot/Sprite/Background` are source art, not the
runtime input aspect ratio.

Every detection and click target is stored in normalized `[0,1]` canvas
coordinates. Other 16:9 resolutions such as 2560×1440 therefore scale exactly.
For the first deployment, 16:10 and ultrawide windows must letterbox a logical
16:9 viewport; they are not silently stretched.

Vision models may consume a 960×540 downsample for identity/layout detection.
OCR and confidence verification should operate on native-resolution crops.

## Browser capture

Install the capture dependency in the client environment:

```bash
pip install playwright
playwright install chromium
```

Open a persistent headed profile so the user can personally accept terms and
authenticate:

```bash
PYTHONPATH=/path/to/sapai:/path/to/sapai-gym \
python scripts/capture_live_sap.py \
  --output-dir live-captures \
  --profile-dir live-browser-profile \
  --headed \
  --wait-seconds 35
```

The recorder never clicks consent or login controls. Once the persistent
profile is authenticated, the same command can run headlessly and capture a
timestamped sequence with `--count` and `--interval-seconds`.

## Browser bridge and dashboard

The installed user service is the normal way to launch the persistent browser:

```bash
systemctl --user start sap-live-browser.service
systemctl --user status sap-live-browser.service
```

It exposes the same dashboard at:

- Spark-local: `http://127.0.0.1:8770`
- Tailscale: `http://100.97.32.64:8770`

In a Remote SSH VS Code window, either open the local address directly or
forward port 8770 from the Ports panel. No VS Code extension is required.

The page streams the real Unity canvas, exposes pause/single-step/emergency
stop, and shows session metadata. It was installed with manual input enabled,
so clicks on the streamed image are forwarded to Playwright in normalized
canvas coordinates. Parsed state, policy decisions, confidence, and the last
executed gesture appear in the side panel.

The persistent profile is already authenticated. Its low-stakes game password
is retrieved inside the browser process from the credential store and is never
included in HTTP requests or session logs.

## Autonomous Arena loop

The controller supports the complete loop: navigate from the main menu, resume
or create an Arena run, parse the team and shop, perform CPU inference, execute
shop actions, accelerate battles, handle result overlays, and continue to the
next shop. It journals every verified shop state atomically to
`runs/live-agent/current-arena-state.json`, so the real server-side Arena run
can be resumed after the browser closes.

Resume the journaled Arena game:

```bash
./scripts/start_live_arena_agent.sh resume
```

Start a new Arena game instead:

```bash
./scripts/start_live_arena_agent.sh new
```

Set `SAP_LIVE_CHECKPOINT=/absolute/path/to/update-NNNNNN.pt` to use another
compatible v19/v21 checkpoint. The default is the v19 update-600 checkpoint
used during controller validation. In-game Settings must remain **Static** for
Held food, **On** for automatic team names, and at **+200%** fast-forward.

The controller maintains recurrent policy state within its process. It pauses
to observe after rolls and level-up rewards, recognizes frozen items after
their visual reordering, and uses verified drag/click gestures for buy, food,
move, combine, freeze, sell, and end-turn actions. During battle it enables
Autoplay/Fast and repeatedly clicks a safe upper-center location until the next
shop or game-over screen appears. A Team Wood desync overlay is reloaded and
the server-side Arena state is then resumed.

To stop the visual client and return CPU resources to training:

```bash
docker stop sap-live-arena-agent 2>/dev/null || true
systemctl --user stop sap-live-browser.service
```

## Recognition assets

The controller uses:

- `runs/live-vision/pet-mobilenet-v1.onnx` for pet identity;
- exact sprite matching for foods;
- `runs/live-vision/stat-digits-v1.npz` for attack/health OCR;
- visual level/progress bars for exact level and experience;
- temporal matching for frozen slots and stochastic shop reconciliation.

The state schema also retains perks, per-turn ability counters, Canned Food
shop buffs, freeze-toggle history, temporary-stat flags, and the last observed
opponent board where available.

## Remaining robustness work

The canonical 1920×1080 Arena profile is playable. More real frames would still
improve robustness for:

- shop/team geometry across turns and shop sizes;
- attack/health digits, levels, experience, gold, hearts, trophies, and turn;
- frozen items and linked level-up choices;
- perks and public temporary-stat highlights;
- opponent boards, cosmetics, and different build/battle backgrounds;
- before/after frames for every executable action.

Arena has been exercised through a complete terminal game and a separate fresh
run. Full 1v1 deployment still needs the same loop connected to the opponent
viewer and broader validation across cosmetics and backgrounds. Subsequent
games should archive low-confidence crops for correction without manually
labeling every frame.
