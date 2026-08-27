#!/usr/bin/env python3
"""Download only the 20 videos in the deterministic R4 pilot manifest."""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "analysis/r4_video_download_manifest.json")
    args = parser.parse_args()
    records = json.loads(args.manifest.read_text())
    results = []
    for index, record in enumerate(records, 1):
        target = Path(record["local_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.stat().st_size > 0:
            status = "already_present"
        else:
            temporary = target.with_suffix(target.suffix + ".r4-part")
            request = urllib.request.Request(record["url"], headers={"User-Agent": "R4-pilot/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
                while block := response.read(4 * 1024 * 1024):
                    stream.write(block)
            if temporary.stat().st_size == 0:
                raise RuntimeError(f"empty download for {record['trajectory_id']}")
            os.replace(temporary, target)
            status = "downloaded"
        results.append({"trajectory_id": record["trajectory_id"], "path": str(target), "bytes": target.stat().st_size, "status": status})
        print(f"[{index}/{len(records)}] {record['trajectory_id']} {status} {target.stat().st_size}", flush=True)
    output = {
        "schema_version": "r4-v1",
        "scope_count": len(records),
        "downloaded": sum(x["status"] == "downloaded" for x in results),
        "already_present": sum(x["status"] == "already_present" for x in results),
        "total_bytes": sum(x["bytes"] for x in results),
        "records": results,
    }
    (ROOT / "analysis/r4_video_materialization.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in ("scope_count", "downloaded", "already_present", "total_bytes")}, indent=2))


if __name__ == "__main__":
    main()
