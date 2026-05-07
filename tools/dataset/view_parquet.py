"""View contents of a parquet dataset file.

Usage:
  python tools/dataset/view_parquet.py <parquet_file> [row_index]

Example:
  python tools/dataset/view_parquet.py experiments/my_exp/data/my_dataset/train.parquet
  python tools/dataset/view_parquet.py smoke_tests/wofost_gym/data/wofost_gym_smoke_llm/test.parquet 5
"""

import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/dataset/view_parquet.py <parquet_file> [row_index]")
        print("")
        print("Arguments:")
        print("  parquet_file - Path to a parquet dataset split")
        print("  row_index   - Row to display (default: first 3 rows)")
        print("")
        print("Example:")
        print("  python tools/dataset/view_parquet.py experiments/my_exp/data/my_dataset/train.parquet")
        print("  python tools/dataset/view_parquet.py smoke_tests/wofost_gym/data/wofost_gym_smoke_llm/test.parquet 5")
        sys.exit(1)

    import pandas as pd

    row_idx = int(sys.argv[2]) if len(sys.argv) > 2 else None
    path = Path(sys.argv[1]).expanduser()
    if not path.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        path = repo_root / path

    if not path.exists():
        print(f"Error: File not found: {path}")
        sys.exit(1)

    df = pd.read_parquet(path)
    print(f"File: {path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    if row_idx is not None:
        rows = [row_idx]
    else:
        rows = range(min(3, len(df)))

    for i in rows:
        print(f"\n{'=' * 60}")
        print(f"Row {i}")
        print(f"{'=' * 60}")
        for col in df.columns:
            val = df.iloc[i][col]
            if isinstance(val, (dict, list)):
                val = json.dumps(val, indent=2, ensure_ascii=False)
            print(f"\n[{col}]:\n{val}")


if __name__ == "__main__":
    main()
