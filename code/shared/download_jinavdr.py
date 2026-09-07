#!/usr/bin/env python3
"""Download the public JinaVDR collection into $EVAL_ROOT/JinaVDR (or $JINAVDR_ROOT)."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

from paths import forbid_venv_path

COLLECTION = "jinaai/jinavdr-visual-document-retrieval-684831c022c53b21c313b449"
API = f"https://huggingface.co/api/collections/{COLLECTION}"
EVIE_ROOT = Path(__file__).resolve().parents[2]
_eval = os.environ.get("EVAL_ROOT")
OUTPUT = Path(
    os.environ.get("JINAVDR_ROOT")
    or ((Path(_eval) / "JinaVDR") if _eval else EVIE_ROOT / "data" / "JinaVDR")
)


def main() -> None:
    forbid_venv_path(OUTPUT, "JinaVDR download dir")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(API, timeout=60) as response:
        payload = json.load(response)
    repo_ids = [
        item["id"]
        for item in payload["items"]
        if item.get("type") == "dataset" and item.get("id", "").startswith("jinaai/")
    ]
    print(f"[JinaVDR] {len(repo_ids)} datasets -> {OUTPUT}")
    for index, repo_id in enumerate(repo_ids, 1):
        target = OUTPUT / repo_id.split("/", 1)[1]
        print(f"[JinaVDR] {index}/{len(repo_ids)} {repo_id} -> {target}", flush=True)
        snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=target)
    print("[JinaVDR] done")


if __name__ == "__main__":
    main()
