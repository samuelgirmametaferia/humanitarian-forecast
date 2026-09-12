from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from humanitarian_forecast.location.models.candidate_rank import ConflictCandidateRanker
from humanitarian_forecast.serving.feature_contract import load_feature_contract
from humanitarian_forecast.serving.inference import OnnxEnsemble, _softmax, greedy_order

DEFAULT_ROWS = (121849, 121850, 157878, 157899)


def _model(sub: dict[str, Any]) -> ConflictCandidateRanker:
    config = sub["model_config"]
    model = ConflictCandidateRanker(**config)
    model.load_state_dict(sub["model_state"])
    return model.eval()


def _contract(data: Any, index: int) -> dict[str, Any]:
    metadata = json.loads(str(data["meta"][index]))
    cutoff = str(
        np.datetime64(metadata["target_date"]) - np.timedelta64(int(metadata["gap_days"]), "D")
    )
    candidates = []
    for candidate_index in range(data["candidate_features"].shape[1]):
        east, north = data["candidate_coordinates"][index, candidate_index]
        candidates.append(
            {
                "id": f"row-{index}-candidate-{candidate_index}",
                "valid": bool(data["candidate_valid"][index, candidate_index]),
                "eastKm": float(east * 1000.0),
                "northKm": float(north * 1000.0),
                "features": data["candidate_features"][index, candidate_index]
                .astype(float)
                .tolist(),
            }
        )
    return {
        "schemaVersion": "feature-contract.v1",
        "country": "Ethiopia",
        "conflictId": str(metadata["conflict_id"]),
        "conflictLabel": str(metadata["conflict"]),
        "anchor": {"latitude": metadata["anchor_lat"], "longitude": metadata["anchor_lon"]},
        "observationCutoff": f"{cutoff}T00:00:00Z",
        "horizonDays": int(metadata["gap_days"]),
        "events": data["x"][index].astype(float).tolist(),
        "candidates": candidates,
    }


def run_parity(
    checkpoint: Path,
    model_dir: Path,
    data_path: Path,
    rows: tuple[int, ...] = DEFAULT_ROWS,
) -> dict[str, Any]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    models = [_model(sub) for sub in state["models"]]
    ensemble = OnnxEnsemble(model_dir)
    data = np.load(data_path, allow_pickle=True, mmap_mode="r")
    logit_tolerance = float(ensemble.manifest["tolerances"]["logitAbsolute"])
    probability_tolerance = float(ensemble.manifest["tolerances"]["probabilityAbsolute"])
    results = []
    passed = True
    for index in rows:
        contract = load_feature_contract(_contract(data, index))
        valid = contract.candidate_valid
        tensors = (
            torch.from_numpy(contract.events[np.newaxis]),
            torch.from_numpy(contract.candidate_features[np.newaxis]),
            torch.from_numpy(valid[np.newaxis]),
            torch.tensor([int(state["country_map"].get(contract.country, 0))]),
            torch.tensor([int(state["conflict_map"].get(contract.conflict_id, 0))]),
        )
        with torch.no_grad():
            torch_logits = tuple(model(*tensors)[0].numpy() for model in models)
        torch_probabilities = np.mean([_softmax(logits, valid) for logits in torch_logits], axis=0)
        onnx_prediction = ensemble.predict(contract)
        maximum_logit_delta = max(
            float(np.max(np.abs(expected - actual)))
            for expected, actual in zip(torch_logits, onnx_prediction.member_logits, strict=True)
        )
        maximum_probability_delta = float(
            np.max(np.abs(torch_probabilities - onnx_prediction.probabilities))
        )
        selection = state.get("selection", {"method": "plain", "top_k": 5})
        masked = np.where(valid, torch_probabilities, 0.0)
        if selection.get("method") == "greedy_spread":
            torch_order = greedy_order(
                masked,
                contract.candidate_coordinates * 1000.0,
                int(selection.get("top_k", 5)),
                float(selection.get("spread_km", 30.0)),
            )
        else:
            torch_order = [
                int(candidate_index)
                for candidate_index in np.argsort(-masked)
                if valid[candidate_index]
            ][: int(selection.get("top_k", 5))]
        onnx_order = [candidate.index for candidate in onnx_prediction.candidates]
        row_passed = (
            maximum_logit_delta <= logit_tolerance
            and maximum_probability_delta <= probability_tolerance
            and torch_order == onnx_order
        )
        passed &= row_passed
        results.append(
            {
                "row": index,
                "conflictId": contract.conflict_id,
                "unseenIdentity": onnx_prediction.unseen_identity,
                "maximumLogitDelta": maximum_logit_delta,
                "maximumProbabilityDelta": maximum_probability_delta,
                "topKOrder": onnx_order,
                "passed": row_passed,
            }
        )
    report = {"schema": "ethiopia-onnx-parity.v1", "rows": results, "passed": passed}
    (model_dir / "parity-report.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest_path = model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "parity-validated" if passed else "parity-failed"
    manifest["parityReport"] = "parity-report.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fine-v2 PyTorch and ONNX predictions")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/location/theswarm/fine_v2/model.pt"),
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models/location/ethiopia_serving"))
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/location/conflict_candidates_64_spillover_v6.npz"),
    )
    parser.add_argument("--rows", type=int, nargs="*", default=list(DEFAULT_ROWS))
    args = parser.parse_args()
    report = run_parity(args.checkpoint, args.model_dir, args.data, tuple(args.rows))
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
