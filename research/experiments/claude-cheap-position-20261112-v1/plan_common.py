"""Shared setup: the exact reference engine/actor stack used by the SH193+ strength evaluations."""
import hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # /workspace inside the container
BASE = ROOT / 'runs/search-transfer-24h-20260916-v1'
OUT = HERE / 'out'

# The current best checkpoint (latest operational parent; ~221M decisions) and the evaluation anchor.
CURRENT = dict(path='runs/ordinary-rl-shared-20260924-v2/outputs/chunk-20/update-025772.pt',
               sha='0669109e9b6ee2d4f0309a3fec83039fc8a7d74f74e3c9f629d2f7026e515d49')


def install(cuda=True):
    """Same import chain as search-iterated-campaign/support.install (reference engine + actor code)."""
    sys.path.insert(0, str(BASE))
    if cuda:
        from common import install as gpu_install  # BASE/common.py; asserts CUDA, FP32, no TF32
        return gpu_install()
    path = ROOT / 'runs/search-played-teaching-20260912-v1/bootstrap.py'
    import importlib.util
    spec = importlib.util.spec_from_file_location('sh187_reference', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.install()


def load_actor(meta=CURRENT, device='cuda'):
    from teacher_batch import BatchedActor
    return BatchedActor.load(ROOT / meta['path'], meta['sha'], device=device)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=1)); tmp.replace(path)


def seed_of(*parts):
    """Deterministic 62-bit seed from a namespace (keeps library/evaluation seeds disjoint)."""
    return int.from_bytes(hashlib.blake2b('|'.join(map(str, parts)).encode(), digest_size=8).digest(), 'big') >> 2
