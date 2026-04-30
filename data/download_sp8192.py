"""
Direct downloader for SP8192 data from kevclark/parameter-golf.

Usage:
    python data/download_sp8192.py --train-shards 1     # smoke test (1 train shard)
    python data/download_sp8192.py --train-shards 143   # full dataset

Files land in:
    data/tokenizers/fineweb_8192_bpe.{model,vocab}
    data/datasets/fineweb10B_sp8192/fineweb_val_000000.bin
    data/datasets/fineweb10B_sp8192/fineweb_train_NNNNNN.bin  (N shards)

Existing files are skipped. SP4096 data is not touched.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path, PurePosixPath

from huggingface_hub import hf_hub_download

REPO_ID = "kevclark/parameter-golf"
REPO_DATASET_PREFIX = "datasets/datasets/fineweb10B_sp8192"
REPO_TOKENIZER_PREFIX = "datasets/tokenizers"

ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets" / "fineweb10B_sp8192"
TOKENIZERS_DIR = ROOT / "tokenizers"


def _download(repo_path: str, local_path: Path) -> None:
    if local_path.exists():
        print(f"  skip (exists): {local_path.relative_to(ROOT.parent)}")
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    p = PurePosixPath(repo_path)
    src = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=p.name,
            subfolder=str(p.parent),
            repo_type="dataset",
        )
    ).resolve()
    shutil.copy2(src, local_path)
    size_mb = local_path.stat().st_size / 1024 / 1024
    print(f"  downloaded ({size_mb:.1f} MB): {local_path.relative_to(ROOT.parent)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SP8192 data from kevclark/parameter-golf")
    parser.add_argument(
        "--train-shards",
        type=int,
        default=1,
        help="Number of training shards to download (default: 1). Max: 143.",
    )
    args = parser.parse_args()

    if args.train_shards < 0 or args.train_shards > 143:
        raise ValueError(f"--train-shards must be 0-143, got {args.train_shards}")

    print(f"Downloading SP8192 from {REPO_ID}")
    print(f"  tokenizer + val shard + {args.train_shards} train shard(s)\n")

    for fname in ("fineweb_8192_bpe.model", "fineweb_8192_bpe.vocab"):
        _download(f"{REPO_TOKENIZER_PREFIX}/{fname}", TOKENIZERS_DIR / fname)

    _download(
        f"{REPO_DATASET_PREFIX}/fineweb_val_000000.bin",
        DATASETS_DIR / "fineweb_val_000000.bin",
    )

    for i in range(args.train_shards):
        fname = f"fineweb_train_{i:06d}.bin"
        _download(f"{REPO_DATASET_PREFIX}/{fname}", DATASETS_DIR / fname)

    print("\nDone.")


if __name__ == "__main__":
    main()
