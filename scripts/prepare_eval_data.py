"""
Extract first 100 en->ga pairs from Dolly_15K_En_Ga.jsonl and save as parquet.
Each line in the source file is a raw Gemini API response object.
"""
import json
import argparse
import os
import pandas as pd


DEFAULT_SRC = r"D:\VS-code-projects\droughts\LoRA_Ga\LoRA_Ga\Dolly_15K_En_Ga.jsonl"
DEFAULT_OUT = "data/eval/dolly_eval_100.parquet"


def extract_pair(line: str) -> dict:
    obj = json.loads(line)
    en = json.loads(obj["request"]["contents"][0]["parts"][0]["text"])
    ga = json.loads(obj["response"]["candidates"][0]["content"]["parts"][0]["text"])
    return {
        "instruction_en": en["instruction"],
        "context_en": en.get("context", ""),
        "response_en": en["response"],
        "instruction_ga": ga["instruction"],
        "context_ga": ga.get("context", ""),
        "response_ga": ga["response"],
        "category": en["category"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=DEFAULT_SRC)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    with open(args.src, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            rows.append(extract_pair(line.strip()))

    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)
    print(f"Saved {len(df)} rows -> {args.out}")
    print(df[["instruction_en", "instruction_ga"]].head(3).to_string())


if __name__ == "__main__":
    main()
