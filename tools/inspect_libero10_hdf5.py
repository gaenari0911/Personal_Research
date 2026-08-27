#!/usr/bin/env python3
"""Read-only structural and integrity inspection for LIBERO-10 HDF5 files."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def dataset_schema(group):
    result = {}

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            result[name] = {"shape": list(obj.shape), "dtype": str(obj.dtype)}

    group.visititems(visitor)
    return result


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_file(path, include_sha256=False):
    report = {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "open_ok": False,
        "errors": [],
    }
    if include_sha256:
        report["sha256"] = sha256_file(path)
    try:
        with h5py.File(path, "r") as handle:
            report["root_keys"] = list(handle.keys())
            if "data" not in handle:
                raise ValueError("missing root group 'data'")
            data = handle["data"]
            report["data_attrs"] = {
                key: jsonable(value)
                for key, value in data.attrs.items()
                if key != "model_file"
            }
            demos = sorted(
                (key for key in data.keys() if key.startswith("demo_")),
                key=lambda value: int(value.rsplit("_", 1)[1]),
            )
            report["demo_count"] = len(demos)
            report["demo_ids"] = demos
            lengths = []
            for demo_id in demos:
                demo = data[demo_id]
                if "actions" not in demo:
                    report["errors"].append(f"{demo_id}: missing actions")
                    continue
                if demo["actions"].ndim != 2 or demo["actions"].shape[1] != 7:
                    report["errors"].append(
                        f"{demo_id}: actions shape {demo['actions'].shape} is not (T, 7)"
                    )
                length = int(demo["actions"].shape[0])
                lengths.append(length)
                for required in ("states", "rewards", "dones", "robot_states"):
                    if required not in demo:
                        report["errors"].append(f"{demo_id}: missing {required}")
                    elif int(demo[required].shape[0]) != length:
                        report["errors"].append(
                            f"{demo_id}: {required} length {demo[required].shape[0]} != {length}"
                        )
                if "obs" not in demo:
                    report["errors"].append(f"{demo_id}: missing obs")
                else:
                    for key, dataset in demo["obs"].items():
                        if int(dataset.shape[0]) != length:
                            report["errors"].append(
                                f"{demo_id}: obs/{key} length {dataset.shape[0]} != {length}"
                            )
            report["length_sum"] = int(sum(lengths))
            report["declared_total"] = int(data.attrs.get("total", -1))
            report["declared_num_demos"] = int(data.attrs.get("num_demos", -1))
            if report["declared_total"] != report["length_sum"]:
                report["errors"].append(
                    f"declared total {report['declared_total']} != length sum {report['length_sum']}"
                )
            if report["declared_num_demos"] != report["demo_count"]:
                report["errors"].append(
                    "declared num_demos "
                    f"{report['declared_num_demos']} != demo count {report['demo_count']}"
                )
            if demos:
                sample = data[demos[0]]
                report["sample_demo"] = demos[0]
                report["sample_demo_attrs"] = {
                    key: (f"<{len(value)} chars>" if key == "model_file" else jsonable(value))
                    for key, value in sample.attrs.items()
                }
                report["sample_schema"] = dataset_schema(sample)
                actions = sample["actions"][()]
                report["sample_action_summary"] = {
                    "min": np.min(actions, axis=0).tolist(),
                    "max": np.max(actions, axis=0).tolist(),
                    "mean": np.mean(actions, axis=0).tolist(),
                    "std": np.std(actions, axis=0).tolist(),
                }
                rewards = sample["rewards"][()]
                dones = sample["dones"][()]
                report["sample_reward_unique"] = np.unique(rewards).tolist()
                report["sample_done_unique"] = np.unique(dones).tolist()
            report["open_ok"] = True
    except Exception as exc:  # preserve all failures in one machine-readable report
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--sha256", action="store_true", help="also stream each file to compute SHA-256"
    )
    args = parser.parse_args()

    files = sorted(args.dataset_dir.glob("*.hdf5"))
    result = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "file_count": len(files),
        "files": [inspect_file(path, args.sha256) for path in files],
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if len(files) != 10 or any(
        (not item["open_ok"]) or item["errors"] for item in result["files"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
