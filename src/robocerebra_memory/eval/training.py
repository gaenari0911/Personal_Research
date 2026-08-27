"""CPU-oriented training and validation selection for frozen Stage B caches."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterable

import torch

from .probes import CONTROL_NAME, TARGET_NAMES, ProbeBank, probe_loss
from .representation_extractor import (
    SCHEMA,
    checkpoint_identity_matches,
    combined_sampling_sha256,
    validate_representation_shard,
)


TARGET_COLUMNS = {
    "current": "gt_current",
    "prev1": "gt_prev1",
    "prev2": "gt_prev2",
    "prev3": "gt_prev3",
    CONTROL_NAME: "gt_current",
}


def load_shards(directory: str | Path, expected_split: str, expected_variant: str) -> list[dict]:
    root = Path(directory)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Stage B representation manifest missing: {manifest_path}")
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SCHEMA
        or manifest.get("split") != expected_split
        or manifest.get("variant") != expected_variant
        or manifest.get("status") != "COMPLETE"
    ):
        raise RuntimeError("Stage B representation manifest contract mismatch")
    if manifest.get("trajectory_count") != manifest.get("expected_trajectory_count"):
        raise RuntimeError("Stage B representation manifest is not complete")
    filenames = manifest.get("shards")
    if not isinstance(filenames, list) or len(set(filenames)) != len(filenames):
        raise RuntimeError("Stage B representation manifest has invalid/duplicate shards")
    payloads = []
    trajectory_ids = set()
    trajectory_hashes = []
    for filename in filenames:
        payload = torch.load(root / filename, map_location="cpu", weights_only=False)
        validate_representation_shard(payload)
        if payload["split"] != expected_split or payload["variant"] != expected_variant:
            raise RuntimeError("Stage B shard split/variant mismatch")
        if payload["trajectory_id"] in trajectory_ids:
            raise RuntimeError("Stage B representation cache has duplicate trajectory IDs")
        trajectory_ids.add(payload["trajectory_id"])
        trajectory_hashes.append((payload["trajectory_id"], payload["sampling_sha256"]))
        manifest_checkpoint = manifest.get("checkpoint", {})
        if not checkpoint_identity_matches(payload, manifest_checkpoint):
            raise RuntimeError("Stage B shard/manifest checkpoint mismatch")
        payloads.append(payload)
    if len(payloads) != manifest["trajectory_count"]:
        raise RuntimeError("Stage B representation shard count mismatch")
    if manifest.get("sampling_sha256") != combined_sampling_sha256(trajectory_hashes):
        raise RuntimeError("Stage B manifest sampling identity hash mismatch")
    return payloads


def _target_tensor(payload: dict, name: str) -> torch.Tensor:
    return torch.tensor(
        [int(sample[TARGET_COLUMNS[name]]) for sample in payload["samples"]], dtype=torch.long
    )


def _representations(payload: dict, name: str) -> torch.Tensor:
    return payload["r_t"].float() if name == CONTROL_NAME else payload["z_t"].float()


def mean_validation_loss(
    probe, payloads: Iterable[dict], name: str, temperature: float
) -> float:
    losses = []
    probe.eval()
    with torch.no_grad():
        for payload in payloads:
            targets = _target_tensor(payload, name)
            if not bool((targets >= 0).any()):
                continue
            loss = probe_loss(
                probe,
                _representations(payload, name),
                payload["candidate_embeddings"].float(),
                targets,
                payload["normalized_candidate_texts"],
                temperature,
            )
            losses.append(float(loss))
    if not losses:
        raise RuntimeError(f"validation has no eligible samples for {name}")
    # Stage B implementation choice: trajectory-macro validation loss.
    return sum(losses) / len(losses)


def fit_probe_bank(
    train_payloads: list[dict],
    val_payloads: list[dict],
    *,
    epochs: int = 20,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    temperature: float = 1.0,
    patience: int = 5,
    seed: int = 42,
) -> tuple[ProbeBank, dict]:
    if not train_payloads or not val_payloads:
        raise ValueError("train and val representation caches are both required")
    if epochs <= 0 or patience <= 0 or learning_rate <= 0 or weight_decay < 0 or temperature <= 0:
        raise ValueError("invalid Stage B probe training hyperparameter")
    torch.manual_seed(seed)
    bank = ProbeBank(seed=seed)
    bank.assert_independent()
    selection: dict[str, dict] = {}
    for name in TARGET_NAMES + (CONTROL_NAME,):
        probe = bank.probes[name]
        optimizer = torch.optim.AdamW(
            probe.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        best_loss = float("inf")
        best_state = None
        best_epoch = 0
        stale = 0
        history = []
        for epoch in range(1, epochs + 1):
            probe.train()
            train_losses = []
            generator = torch.Generator().manual_seed(seed + epoch)
            for index in torch.randperm(len(train_payloads), generator=generator).tolist():
                payload = train_payloads[index]
                targets = _target_tensor(payload, name)
                if not bool((targets >= 0).any()):
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss = probe_loss(
                    probe,
                    _representations(payload, name),
                    payload["candidate_embeddings"].float(),
                    targets,
                    payload["normalized_candidate_texts"],
                    temperature,
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite Stage B probe loss for {name}")
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach()))
            validation_loss = mean_validation_loss(probe, val_payloads, name, temperature)
            history.append(
                {
                    "epoch": epoch,
                    "train_trajectory_macro_loss": sum(train_losses) / len(train_losses),
                    "val_trajectory_macro_loss": validation_loss,
                }
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(probe.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is None:
            raise RuntimeError(f"no Stage B probe selected for {name}")
        probe.load_state_dict(best_state, strict=True)
        probe.eval()
        selection[name] = {
            "best_epoch": best_epoch,
            "best_val_trajectory_macro_loss": best_loss,
            "epochs_run": len(history),
            "history": history,
        }
    return bank, {
        "schema_version": "stage-b-probes-v1",
        "selection_split": "val",
        "backbone_frozen": True,
        "probe_names": list(TARGET_NAMES + (CONTROL_NAME,)),
        "independent_parameters": True,
        "same_current_z_for_temporal_probes": True,
        "seed": seed,
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "temperature": temperature,
        "max_epochs": epochs,
        "patience": patience,
        "selection": selection,
    }
