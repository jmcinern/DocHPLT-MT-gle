"""Agent 1 (Models) — download HPLT Marian models, convert to HuggingFace
MarianMTModel format, store them, and sanity-check a translation.

HPLT publishes en->ga and ga->en as raw Marian binaries (.npz + .spm + .vocab)
and does NOT ship the decoder.yml / vocab.yml that HF's converter expects, so we
synthesise those from the npz config and the .vocab file before converting.

Usage:
    python models/setup_models.py            # both models
    python models/setup_models.py en-ga      # one model
"""

import sys
import shutil
from pathlib import Path

import yaml
import torch
from huggingface_hub import hf_hub_download
from transformers import MarianMTModel, MarianTokenizer

# Vendored HF converter (not shipped in the installed transformers wheel).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert_marian_to_pytorch as mc  # noqa: E402

HERE = Path(__file__).resolve().parent
STAGE_ROOT = HERE / "_marian_src"      # synthesised Marian source dirs
OUT_ROOT = HERE / "converted"          # final HF models

NPZ_FILE = "model.npz.best-chrf.npz"
BEAM_SIZE = 6  # only field the converter reads from decoder.yml

MODELS = {
    "en-ga": {
        "repo": "HPLT/translate-en-ga-v2.0-hplt",
        "spm": "model.en-ga.spm",
        "vocab": "model.en-ga.vocab",
        "test_src": "Translate this sentence into Irish, please.",
    },
    "ga-en": {
        "repo": "HPLT/translate-ga-en-v2.0-hplt",
        "spm": "model.ga-en.spm",
        "vocab": "model.ga-en.vocab",
        "test_src": "Aistrigh an abairt seo go Béarla, le do thoil.",
    },
}


def vocab_to_yaml(vocab_path: Path, out_path: Path) -> int:
    """Convert a Marian/SentencePiece .vocab ('<token>\\t<score>' per line, id =
    line index) into the token->id YAML mapping the HF converter expects."""
    mapping = {}
    with open(vocab_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.rstrip("\n")
            if not line:
                continue
            token = line.rsplit("\t", 1)[0]
            if token in mapping:
                raise ValueError(f"Duplicate vocab token {token!r} at id {idx}")
            mapping[token] = idx
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(mapping, f, allow_unicode=True, sort_keys=False)
    return len(mapping)


def stage_marian_source(tag: str, cfg: dict) -> Path:
    """Download HPLT files and assemble a Marian source dir the converter accepts."""
    repo = cfg["repo"]
    print(f"[{tag}] downloading from {repo} ...")
    npz = hf_hub_download(repo_id=repo, filename=NPZ_FILE)
    spm = hf_hub_download(repo_id=repo, filename=cfg["spm"])
    vocab = hf_hub_download(repo_id=repo, filename=cfg["vocab"])

    src_dir = STAGE_ROOT / tag
    if src_dir.exists():
        shutil.rmtree(src_dir)
    src_dir.mkdir(parents=True)

    shutil.copy(npz, src_dir / NPZ_FILE)
    # tied (shared) vocab -> source and target spm are the same file
    shutil.copy(spm, src_dir / "source.spm")
    shutil.copy(spm, src_dir / "target.spm")

    n = vocab_to_yaml(Path(vocab), src_dir / "vocab.yml")
    print(f"[{tag}] vocab.yml written ({n} tokens)")

    with open(src_dir / "decoder.yml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"beam-size": BEAM_SIZE}, f)

    return src_dir


def convert(tag: str, src_dir: Path) -> Path:
    """Convert staged Marian source -> HF MarianMTModel, saved in fp32."""
    dest_dir = OUT_ROOT / f"{tag}-hplt-hf"
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)

    state = mc.OpusState(src_dir)
    state.tokenizer.save_pretrained(dest_dir)
    model = state.load_marian_model()  # fp32; HF's convert() would .half() here
    model.save_pretrained(dest_dir)
    print(f"[{tag}] saved HF model -> {dest_dir}")
    return dest_dir


def sanity_check(tag: str, dest_dir: Path, test_src: str) -> None:
    """Reload from disk and run one translation to prove it loads and generates."""
    tok = MarianTokenizer.from_pretrained(dest_dir)
    model = MarianMTModel.from_pretrained(dest_dir)
    model.eval()
    batch = tok([test_src], return_tensors="pt", padding=True)
    with torch.no_grad():
        gen = model.generate(**batch, max_length=128, num_beams=BEAM_SIZE)
    out = tok.batch_decode(gen, skip_special_tokens=True)[0]
    params = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] params={params:,} dtype={next(model.parameters()).dtype}")
    print(f"[{tag}] SRC: {test_src}")
    print(f"[{tag}] OUT: {out}")
    if not out.strip():
        raise RuntimeError(f"[{tag}] empty translation output")


def main(tags):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for tag in tags:
        cfg = MODELS[tag]
        src_dir = stage_marian_source(tag, cfg)
        dest_dir = convert(tag, src_dir)
        sanity_check(tag, dest_dir, cfg["test_src"])
        print(f"[{tag}] OK\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    tags = args if args else list(MODELS)
    unknown = [t for t in tags if t not in MODELS]
    if unknown:
        raise SystemExit(f"Unknown model tag(s): {unknown}. Choose from {list(MODELS)}")
    main(tags)
