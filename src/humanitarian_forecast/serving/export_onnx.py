from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn

from humanitarian_forecast.location.models.candidate_rank import ConflictCandidateRanker

INPUT_NAMES = ["events", "candidates", "valid", "country", "conflict"]
OUTPUT_NAMES = ["logits"]


class ExportableRanker(nn.Module):
    def __init__(self, model: ConflictCandidateRanker) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        events: torch.Tensor,
        candidates: torch.Tensor,
        valid: torch.Tensor,
        country: torch.Tensor,
        conflict: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(events, candidates, valid, country, conflict)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model(sub: dict[str, Any]) -> ConflictCandidateRanker:
    config = sub["model_config"]
    model = ConflictCandidateRanker(
        config["event_dim"],
        config["candidate_dim"],
        config["sequence_length"],
        config["countries"],
        config["conflicts"],
        config["d_model"],
        config["heads"],
        config["layers"],
        config["ff_dim"],
        config["dropout"],
    )
    model.load_state_dict(sub["model_state"])
    return model.eval()


def export_package(checkpoint: Path, output: Path, opset: int = 20) -> dict[str, Any]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    output.mkdir(parents=True, exist_ok=True)
    source_info = checkpoint.with_name("info.blt")
    if source_info.exists():
        shutil.copyfile(source_info, output / "champion-info.json")
    sample = (
        torch.zeros((1, 16, 19), dtype=torch.float32),
        torch.zeros((1, 64, 37), dtype=torch.float32),
        torch.ones((1, 64), dtype=torch.bool),
        torch.zeros((1,), dtype=torch.int64),
        torch.zeros((1,), dtype=torch.int64),
    )
    member_files = []
    for index, sub in enumerate(state["models"]):
        destination = output / f"member-{index}.onnx"
        torch.onnx.export(
            ExportableRanker(_model(sub)),
            sample,
            destination,
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            opset_version=opset,
            dynamo=False,
            external_data=False,
            do_constant_folding=True,
        )
        member_files.append({"path": destination.name, "sha256": _sha256(destination)})
    manifest = {
        "schema": "ethiopia-serving-package.v1",
        "status": "parity-pending",
        "model": "theswarm_fine_v2",
        "featureContract": "feature-contract.v1",
        "sourceCheckpointSha256": _sha256(checkpoint),
        "members": member_files,
        "countryMap": {str(key): int(value) for key, value in state["country_map"].items()},
        "conflictMap": {str(key): int(value) for key, value in state["conflict_map"].items()},
        "selection": state.get("selection", {"method": "plain", "top_k": 5}),
        "input": {"events": [16, 19], "candidates": [64, 37]},
        "tolerances": {"logitAbsolute": 0.0001, "probabilityAbsolute": 0.000001},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the unchanged fine-v2 ensemble to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/location/theswarm/fine_v2/model.pt"),
    )
    parser.add_argument("--output", type=Path, default=Path("models/location/ethiopia_serving"))
    parser.add_argument("--opset", type=int, default=20)
    args = parser.parse_args()
    manifest = export_package(args.checkpoint, args.output, args.opset)
    print(json.dumps({"members": len(manifest["members"]), "status": manifest["status"]}))


if __name__ == "__main__":
    main()
