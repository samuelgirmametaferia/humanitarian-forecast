"""Typed serving bridge for the parity-validated Ethiopia ranker."""

from .feature_contract import CandidateInput, FeatureContract, Position, load_feature_contract
from .inference import CandidatePrediction, EnsemblePrediction, OnnxEnsemble

__all__ = [
    "CandidateInput",
    "CandidatePrediction",
    "EnsemblePrediction",
    "FeatureContract",
    "OnnxEnsemble",
    "Position",
    "load_feature_contract",
]
