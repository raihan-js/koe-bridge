"""Upload eval artifacts to HF dataset raihan-js/koe-bridge-eval.

Contents: per-row M1 cascade rows (both dirs), M1/M2/M3 summary JSONs.
No audio, no model weights, no transcripts beyond short FLEURS captions
used for scoring (FLEURS is CC-BY-4.0; rows carry IDs + ASR/MT text).
Re-run any time; files overwritten with newer results.
"""
from pathlib import Path

from huggingface_hub import HfApi

REPO = "koe-bridge-eval"
FILES = [
    "results/m1_rows_ja_jp.jsonl",
    "results/m1_rows_en_us.jsonl",
    "results/m1.json",
    "results/m2.json",
    "results/m3.json",
    "results/neg.json",
]

CARD = """---
license: cc-by-4.0
---
# koe-bridge-eval

Evaluation artifacts for [koe-bridge](https://github.com/raihan-js/koe-bridge)
(open streaming Japanese-English speech translator, all-local pipeline).

- `m1_rows_*.jsonl`: per-utterance cascade rows (FLEURS ID, reference,
  ASR text, MT text, roundtrip text). Source audio NOT included.
- `m1.json`: cascade audit summary (n=200/direction).
- `m2.json`: streaming first-audio lag (n=20/direction).
- `m3.json`: prefix-translation survival / word-order wait cost (n=100/dir).
- `neg.json`: negation-polarity flips on prefixes (n=40/direction).

Source utterances: FLEURS test set (CC-BY-4.0, Google).
"""


def main() -> None:
    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", exist_ok=True)
    root = Path(__file__).resolve().parent.parent
    for f in FILES:
        api.upload_file(path_or_fileobj=str(root / f), path_in_repo=f,
                        repo_id=f"raihan-js/{REPO}", repo_type="dataset")
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                    repo_id=f"raihan-js/{REPO}", repo_type="dataset")
    print(f"uploaded {len(FILES)} files + card to raihan-js/{REPO}")


if __name__ == "__main__":
    main()
