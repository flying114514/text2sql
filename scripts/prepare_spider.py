"""Prepare the Spider dev set locally from the HF dataset `premai-io/spider`.

Layout on the hub:
    validation.json                       -> the dev split (questions + gold SQL)
    database/<db_id>/<db_id>.sqlite       -> the executable databases

We only need the dev split and the ~20 databases it references (not all 169),
which keeps the download small and robust over a flaky proxy. Every network
call is retried.

    uv run python scripts/prepare_spider.py            # step A: dev.json only + report
    uv run python scripts/prepare_spider.py --dbs      # step B: also download dev databases
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "premai-io/spider"
ROOT = Path(__file__).resolve().parents[1]
SPIDER_DIR = ROOT / "data" / "spider"


def with_retry(fn, attempts: int = 8, delay: float = 3.0):
    last = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"    retry {i}/{attempts}: {type(e).__name__}")
            time.sleep(delay)
    raise last


def fetch(remote_path: str) -> Path:
    """Download one file from the repo into data/spider/, with retries."""
    local = with_retry(
        lambda: hf_hub_download(
            repo_id=REPO,
            repo_type="dataset",
            filename=remote_path,
            local_dir=str(SPIDER_DIR),
        )
    )
    return Path(local)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbs", action="store_true", help="also download the dev databases")
    parser.add_argument(
        "--train",
        action="store_true",
        help="also download the TRAIN split (few-shot example pool; text only, no databases)",
    )
    args = parser.parse_args()

    SPIDER_DIR.mkdir(parents=True, exist_ok=True)

    # Step A: the dev split.
    print("Downloading validation.json (dev split) ...")
    dev_file = fetch("validation.json")
    # Standardise the name our loader expects.
    dev_json = SPIDER_DIR / "dev.json"
    dev_json.write_bytes(dev_file.read_bytes())

    rows = json.loads(dev_json.read_text(encoding="utf-8"))
    sample = rows[0]
    db_ids = sorted({r["db_id"] for r in rows})
    print(f"\ndev examples : {len(rows)}")
    print(f"distinct dbs : {len(db_ids)}")
    print(f"sample keys  : {list(sample.keys())}")
    print(f"sample row   : db_id={sample.get('db_id')!r}")
    print(f"               question={sample.get('question')!r}")
    print(f"               query={sample.get('query')!r}")
    print(f"db_ids       : {db_ids}")

    # Optional: the train split, used purely as a few-shot demonstration pool.
    # Spider's train databases are DISJOINT from dev, so train examples never
    # leak dev schemas/answers — the proper, leakage-free way to do few-shot.
    if args.train:
        print("\nDownloading TRAIN split (few-shot pool) ...")
        train_rows = None
        for cand in ("train.json", "train_spider.json", "train_others.json"):
            try:
                f = fetch(cand)
                train_rows = json.loads(Path(f).read_text(encoding="utf-8"))
                print(f"  got {cand}: {len(train_rows)} examples")
                break
            except Exception as e:  # noqa: BLE001
                print(f"  {cand} not available ({type(e).__name__})")
        if train_rows is not None:
            train_json = SPIDER_DIR / "train.json"
            train_json.write_text(json.dumps(train_rows, ensure_ascii=False), encoding="utf-8")
            train_dbs = sorted({r["db_id"] for r in train_rows})
            overlap = set(train_dbs) & set(db_ids)
            print(f"  saved -> {train_json}")
            print(f"  train examples: {len(train_rows)}, distinct dbs: {len(train_dbs)}")
            print(f"  train∩dev dbs : {len(overlap)} (should be 0 = no leakage)")
        else:
            print("  [WARN] could not fetch a train split; few-shot pool not created.")

    if not args.dbs:
        print("\n[Step A done] Re-run with --dbs to download the databases.")
        return

    # Step B: only the databases referenced by the dev set.
    print(f"\nDownloading {len(db_ids)} dev databases ...")
    ok = 0
    for i, db_id in enumerate(db_ids, 1):
        try:
            fetch(f"database/{db_id}/{db_id}.sqlite")
            ok += 1
            print(f"  [{i}/{len(db_ids)}] {db_id}  OK")
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(db_ids)}] {db_id}  FAILED: {type(e).__name__}")
    print(f"\n[Step B done] {ok}/{len(db_ids)} databases downloaded into {SPIDER_DIR / 'database'}")


if __name__ == "__main__":
    main()
