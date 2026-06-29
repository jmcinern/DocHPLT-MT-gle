"""
Download HPLT/DocHPLT en-ga from HuggingFace and save as parquet shards.
Run inside the Qomhra container on LUMI.
"""
import argparse
import os
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/scratch/project_465002364/DocHPLT-MT-gle/data/train")
    parser.add_argument("--split", default="train")
    parser.add_argument("--shard_size", type=int, default=100_000, help="rows per parquet shard")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading HPLT/DocHPLT en-ga ({args.split})...")
    ds = load_dataset("HPLT/DocHPLT", "en-ga", split=args.split, trust_remote_code=True)
    print(f"Dataset loaded: {len(ds)} rows, columns: {ds.column_names}")

    n_shards = max(1, len(ds) // args.shard_size)
    for i in range(n_shards):
        start = i * args.shard_size
        end = start + args.shard_size if i < n_shards - 1 else len(ds)
        shard = ds.select(range(start, end))
        out_path = os.path.join(args.output_dir, f"shard_{i:04d}.parquet")
        shard.to_parquet(out_path)
        print(f"  Shard {i}: rows {start}-{end} -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
