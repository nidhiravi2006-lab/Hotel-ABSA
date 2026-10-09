"""Download safe Parquet data and create a reproducible review sample.

The accessible redistribution has five aspect labels. Service and business
service remain missing; this script never substitutes overall ratings for them.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import sys
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://huggingface.co/api/datasets/jniimi/tripadvisor-review-rating/parquet/default/train/0.parquet"
ASPECTS = ["service", "rooms", "location", "value", "cleanliness", "sleep_quality", "business_service"]


def prepare(source_path, output, sample_size=20000, seed=42):
    raw = pd.read_parquet(source_path)
    frame = pd.DataFrame()
    frame["review_id"] = [f"hf-{i}" for i in raw.index]
    frame["title"] = raw["title"].fillna("")
    frame["text"] = raw["text"].fillna("")
    # Use reviewer groups. The redistributed hotel_id does not establish
    # reliable hotel grouping and is intentionally excluded from the model.
    frame["user_id"] = raw["user_id"].fillna("").astype(str)
    for target in ["overall", *ASPECTS]:
        values = pd.to_numeric(raw[target], errors="coerce") if target in raw else pd.Series(np.nan, index=raw.index)
        frame[target] = values.where(values.between(1, 5) & (values % 1 == 0))
    joined = (frame["title"] + " " + frame["text"]).map(lambda t: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(t))).strip().lower())
    valid = joined.str.len().ge(20) & frame["overall"].notna()
    frame = frame.loc[valid].copy()
    frame["text_hash"] = joined.loc[valid].map(lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest())
    before = len(frame)
    frame = frame.drop_duplicates("text_hash")
    if len(frame) > sample_size:
        frame = frame.sample(n=sample_size, random_state=seed)
    frame = frame.sort_values("review_id").reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    metadata = {
        "dataset": "jniimi/tripadvisor-review-rating",
        "dataset_url": "https://huggingface.co/datasets/jniimi/tripadvisor-review-rating",
        "download_url": SOURCE,
        "original_paper": "https://aclanthology.org/D13-1199/",
        "redistribution_license": "Apache-2.0 as stated by the publisher; consult original source terms",
        "source_rows": len(raw), "rows_after_cleaning_before_deduplication": before,
        "sample_rows": len(frame), "sample_seed": seed,
        "source_parquet_sha256": hashlib.file_digest(open(source_path, "rb"), "sha256").hexdigest(),
        "sample_csv_sha256": hashlib.file_digest(open(output, "rb"), "sha256").hexdigest(),
        "label_counts": {a: int(frame[a].notna().sum()) for a in ["overall", *ASPECTS]},
        "rating_counts": {a: {str(int(k)): int(v) for k, v in frame[a].value_counts().sort_index().items()} for a in ["overall", *ASPECTS]},
        "limitation": "Different dataset from Yu (2016). No service or business_service ratings in this redistribution. No verified hotel grouping. Review ratings are weak labels for text sentiment.",
    }
    output.with_name("provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Use an existing source Parquet file")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "reviews.csv")
    parser.add_argument("--sample-size", type=int, default=20000)
    args = parser.parse_args()
    source = args.input or ROOT / "data" / "raw" / "source.parquet"
    if args.sample_size < 100:
        parser.error("sample-size must be at least 100")
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        partial = source.with_suffix(".part")
        request = urllib.request.Request(SOURCE, headers={"User-Agent": "HotelReviewABSA/1.0"})
        print("Downloading public Parquet data...", flush=True)
        with urllib.request.urlopen(request, timeout=90) as response, open(partial, "wb") as dest:
            while chunk := response.read(1024 * 1024):
                dest.write(chunk)
        partial.replace(source)
    print(json.dumps(prepare(source, args.output, args.sample_size), indent=2))


if __name__ == "__main__":
    main()
