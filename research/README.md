# Latest shared research and models

Download the [research release](https://github.com/PatanL/sapai-gym/releases/tag/research-update-25772-20261112)
for the current policy, the November positioning experiment, and their exact
reference source. The separate [sapai repository](https://github.com/PatanL/sapai)
contains the current development simulator.

## What is included

| Artifact | Meaning |
| --- | --- |
| `update-025772.pt` | Latest completed self-play policy; update 25,772, 221,367,362 recorded decisions; model, optimizer and replay ledger included |
| `battle_net.pt` | Battle predictor used to screen positioning candidates |
| `posnet-zeroshot.pt` | Experimental END-positioner initialized from the battle predictor; not a completed trained or strength-validated replacement |
| `research-workspace.tar.gz` | Source, frozen simulator/actor dependencies, opponent-board library, vocabularies, available positioning labels, measurements, run config and league metadata, in their original workspace paths |
| `RELEASE_MANIFEST.json` | SHA-256 hashes, source commits and model restoration paths |

The newest experiment source is also browsable in
[`experiments/`](experiments/). These are copies for code review: the programs
expect the `runs/` paths in the release workspace when executed.

- [`claude-planner-20261013-v1`](experiments/claude-planner-20261013-v1/README.md):
  shop/position planning around the fixed update-25,772 policy.
- [`claude-cheap-position-20261112-v1`](experiments/claude-cheap-position-20261112-v1/README.md):
  positioning latency and quality measurements.
- [`claude-end-positioner-20261112-v1`](experiments/claude-end-positioner-20261112-v1/):
  a separate learned positioning module, public-label generation and training.

## Restore the workspace

Download all five assets into a new directory. `gh` may use your existing GitHub
login; all release files are also downloadable from the public release page.

```bash
mkdir sap-rl-shared
cd sap-rl-shared
gh release download research-update-25772-20261112 --repo PatanL/sapai-gym
python3 - <<'PY'
import hashlib, json
from pathlib import Path
m = json.loads(Path('RELEASE_MANIFEST.json').read_text())
for name, meta in [(m['archive']['name'], m['archive']), *m['model_assets'].items()]:
    with Path(name).open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == meta['sha256'], name
print('All release asset hashes passed')
PY
tar -xzf research-workspace.tar.gz
python3 - <<'PY'
import json, shutil
from pathlib import Path
m = json.loads(Path('RELEASE_MANIFEST.json').read_text())
for name, meta in m['model_assets'].items():
    target = Path(meta['workspace_path'])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(name, target)
PY
```

The archive provides a source snapshot, not Git authentication or `.git`
directories. Clone the repositories separately if you need their history.

## Runtime and smoke verification

The original experiments used NVIDIA PyTorch `25.09-py3` on DGX Spark. The
original local runtime image ID is recorded in the manifest; an image ID is
not a downloadable container registry address. A portable build recipe is
provided for a fresh runtime:

```bash
docker build -f sapai-gym/research/Dockerfile -t sap-rl-shared .
docker run --rm --network none --cpus 4 --memory 6g \
  -v "$PWD:/workspace:ro" -w /workspace \
  --entrypoint python sap-rl-shared \
  sapai-gym/research/smoke_shared_release.py
```

The smoke check verifies the frozen source hashes, strict policy loading,
finite model weights, one CPU action, and positioner inference. Full training
and the original paired evaluation runs require CUDA. The historical `run.sh`
wrappers refer to the old local image ID; on another machine, invoke Python
inside the new image with `/workspace` mounted instead.

For example, to evaluate the public simulated-opponent positioner:

```bash
docker run --rm --gpus all --cpus 6 --memory 10g --shm-size 2g \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -v "$PWD:/workspace" \
  -w /workspace/runs/claude-end-positioner-20261112-v1 \
  --entrypoint python sap-rl-shared match.py \
  --mode planner --pairs 1 --workers 5 --namespace shared-smoke-v1 \
  --position-only --sim-opponent --out shared-smoke.jsonl
```

`train_posnet.py` trains the separate positioner from `out/labels-*.jsonl`.
Only the partial label collection actually on disk is included; the launcher
requesting 300 games does not imply that 300 games completed.

## Mechanics and continuation caveats

The normal development checkout is revision
`turtle-current-frozen-shop-cap-v28-20260905`. The latest policy and planners
use the separately frozen
`turtle-v30-food-right-reorder10-v1-20260911` simulator and the
`entity-public-reorder-budget-v3-20260911` observation schema. The reference
bootstrap in the archive loads and authenticates those exact dependencies.
Use it when reproducing the checkpoint; using the development checkout by
itself is a different ruleset.

Recorded repairs include frozen-food ordering, the shared ten-slot shop cap,
summon/Tiger event ordering, level-three Chocolate legality, lifetime-safe
freeze/formation identities, and exact clone/history restoration. The archive
also contains the v29/v30 implementation notes. Earlier unresolved questions
in `CORRECTNESS_AUDIT_V18.md` are historical caveats, not evidence that every
edge interaction was subsequently verified. SAP-Calculator is a battle
reference; shop behavior needs live-game evidence.

Optimizer state alone does not reproduce the latest historical learner. Its
trainer refuses to discard a saved privileged-replay ledger on resume. Exact
continuation needs the original replay corpus and fixed/rolling resident
models, which this compact release does not include. For a new training
experiment, use explicit model-only initialization and a fresh optimizer,
counter, league and output directory; do not describe that as an exact resume.
The frozen training source, entrypoints, run configuration and ledger metadata
are included for inspection.

Historical paired evaluations in the experiment READMEs remain their original
measurements. Publication and a successful smoke check do not establish a new
strength improvement or a human-level benchmark result.

## Repackage later work

From the original workspace, use Python 3.11 or newer:

```bash
python3 sapai-gym/scripts/package_research_release.py --export-experiments
python3 sapai-gym/scripts/package_research_release.py --output-dir /path/to/new-release-directory
```

This packages only selected source and research artifacts, checks sealed
reference hashes, screens text for credential markers, and excludes Git
credentials, browser profiles and intermediate checkpoint collections.
