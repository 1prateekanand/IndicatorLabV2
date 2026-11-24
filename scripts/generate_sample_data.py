"""Create SYNTHETIC 1-minute test data (random walk). NOT market data; only for exercising the pipeline.

Usage:  python scripts/generate_sample_data.py --days 60 --out data/samples/SYNTH_1m_demo.csv
Then import the CSV via the API or the Dataset page, or:
  curl -X POST localhost:8000/api/datasets/import -H 'Content-Type: application/json' \
       -d '{"path":"data/samples/SYNTH_1m_demo.csv","name":"SYNTH_1m_demo","symbol":"SYNTHETIC"}'
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from indicatorlab.data import generate_synthetic  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/samples/SYNTH_1m_demo.csv")
    args = parser.parse_args()
    path = generate_synthetic(args.out, days=args.days, seed=args.seed)
    print(f"wrote synthetic data to {path}")
