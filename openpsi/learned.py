"""Contracts and utilities for optional learned appraisal residuals.

This module intentionally has no training-framework dependency.  Runtime
predictors consume a one-dimensional NumPy vector and return normalized
residual coordinates in ``[-1, 1]``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from core.features import StimulusFeatures
from core.state import MotivationalState
from openpsi.profile import AppraisalProfile


class ResidualPredictor(Protocol):
    """Framework-neutral inference contract for learned appraisal."""

    def predict(self, input_vector: np.ndarray) -> np.ndarray:
        """Return one normalized residual per configured output modulator."""


@dataclass(frozen=True)
class ZeroResidualPredictor:
    """Phase-1 predictor used to verify integration without changing state."""

    output_size: int

    def __post_init__(self) -> None:
        if self.output_size <= 0:
            raise ValueError("output_size must be positive")

    def predict(self, input_vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(input_vector, dtype=float)
        if vector.ndim != 1:
            raise ValueError("learned-appraisal input must be one-dimensional")
        if not np.all(np.isfinite(vector)):
            raise ValueError("learned-appraisal input must be finite")
        return np.zeros(self.output_size, dtype=float)


def file_sha256(path: str | Path) -> str:
    """Return a lowercase SHA-256 digest for an artifact."""

    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LearnedAppraisalManifest:
    """Versioned coordinate and artifact contract for one residual model."""

    model_id: str
    application_name: str
    appraisal_profile_name: str
    goal_names: tuple[str, ...]
    modulator_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    output_modulator_names: tuple[str, ...]
    residual_bounds: tuple[float, ...]
    manifest_version: int = 1
    input_mean: tuple[float, ...] = ()
    input_scale: tuple[float, ...] = ()
    artifact_path: str | None = None
    artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        for label in (
            "model_id",
            "application_name",
            "appraisal_profile_name",
        ):
            if not str(getattr(self, label)).strip():
                raise ValueError(f"{label} must be a non-empty string")

        if self.manifest_version != 1:
            raise ValueError(
                f"unsupported learned-appraisal manifest version: {self.manifest_version}"
            )

        for label, names in (
            ("goal_names", self.goal_names),
            ("modulator_names", self.modulator_names),
            ("feature_names", self.feature_names),
            ("output_modulator_names", self.output_modulator_names),
        ):
            if not names:
                raise ValueError(f"{label} must not be empty")
            if len(names) != len(set(names)):
                raise ValueError(f"{label} must contain unique names")
            if any(not isinstance(name, str) or not name.strip() for name in names):
                raise ValueError(f"{label} must contain non-empty strings")

        if len(self.residual_bounds) != len(self.output_modulator_names):
            raise ValueError(
                "residual_bounds must match output_modulator_names length"
            )
        bounds = np.asarray(self.residual_bounds, dtype=float)
        if not np.all(np.isfinite(bounds)) or np.any(bounds < 0.0):
            raise ValueError("residual bounds must be finite and non-negative")

        missing_outputs = set(self.output_modulator_names).difference(
            self.modulator_names
        )
        if missing_outputs:
            raise ValueError(
                f"output modulators are absent from the schema: {sorted(missing_outputs)}"
            )

        if bool(self.input_mean) != bool(self.input_scale):
            raise ValueError("input_mean and input_scale must be provided together")
        if self.input_mean:
            if len(self.input_mean) != self.input_size:
                raise ValueError("input_mean length does not match manifest input size")
            if len(self.input_scale) != self.input_size:
                raise ValueError("input_scale length does not match manifest input size")
            means = np.asarray(self.input_mean, dtype=float)
            scales = np.asarray(self.input_scale, dtype=float)
            if not np.all(np.isfinite(means)):
                raise ValueError("input means must be finite")
            if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
                raise ValueError("input scales must be finite and positive")

        if bool(self.artifact_path) != bool(self.artifact_sha256):
            raise ValueError(
                "artifact_path and artifact_sha256 must be provided together"
            )
        if self.artifact_sha256 is not None:
            digest = self.artifact_sha256.lower()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError("artifact_sha256 must be a 64-character hex digest")

    @property
    def input_size(self) -> int:
        """Size of ``[G, M, features, M_rule, boundary_pressure]``."""

        return (
            len(self.goal_names)
            + len(self.modulator_names)
            + len(self.feature_names)
            + len(self.modulator_names)
            + 1
        )

    @property
    def output_size(self) -> int:
        return len(self.output_modulator_names)

    def validate_runtime(
        self,
        state: MotivationalState,
        profile: AppraisalProfile,
        features: StimulusFeatures,
    ) -> None:
        """Reject schema, profile, or feature-order drift before inference."""

        if tuple(state.schema.goal_names) != self.goal_names:
            raise ValueError("learned-appraisal goal schema does not match manifest")
        if tuple(state.schema.modulator_names) != self.modulator_names:
            raise ValueError("learned-appraisal modulator schema does not match manifest")
        if profile.name != self.appraisal_profile_name:
            raise ValueError("appraisal profile does not match learned manifest")
        missing = [name for name in self.feature_names if name not in features.features]
        if missing:
            raise ValueError(f"learned-appraisal features are missing: {missing}")

    def validate_artifact(self, base_directory: str | Path | None = None) -> None:
        """Verify an optional model artifact against its manifest digest."""

        if self.artifact_path is None:
            return
        artifact = Path(self.artifact_path)
        if not artifact.is_absolute() and base_directory is not None:
            artifact = Path(base_directory) / artifact
        if not artifact.is_file():
            raise ValueError(f"learned-appraisal artifact does not exist: {artifact}")
        actual = file_sha256(artifact)
        if actual != self.artifact_sha256.lower():
            raise ValueError("learned-appraisal artifact checksum mismatch")

    def to_dict(self) -> dict:
        values = asdict(self)
        for name in (
            "goal_names",
            "modulator_names",
            "feature_names",
            "output_modulator_names",
            "residual_bounds",
            "input_mean",
            "input_scale",
        ):
            values[name] = list(values[name])
        return values

    @classmethod
    def from_dict(cls, values: dict) -> "LearnedAppraisalManifest":
        data = dict(values)
        for name in (
            "goal_names",
            "modulator_names",
            "feature_names",
            "output_modulator_names",
            "residual_bounds",
            "input_mean",
            "input_scale",
        ):
            if name in data:
                data[name] = tuple(data[name])
        return cls(**data)

    @classmethod
    def read_json(cls, path: str | Path) -> "LearnedAppraisalManifest":
        manifest_path = Path(path)
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = cls.from_dict(json.load(handle))
        if manifest.artifact_path is not None:
            artifact = Path(manifest.artifact_path)
            if not artifact.is_absolute():
                artifact = (manifest_path.parent / artifact).resolve()
            manifest = cls.from_dict(
                {
                    **manifest.to_dict(),
                    "artifact_path": str(artifact),
                }
            )
        manifest.validate_artifact()
        return manifest


@dataclass(frozen=True)
class ManifestFeatureEncoder:
    """Build the exact, manifest-ordered learned-appraisal input vector."""

    manifest: LearnedAppraisalManifest
    boundary_pressure: Callable[[MotivationalState], float]

    def pressure(self, state: MotivationalState) -> float:
        value = float(self.boundary_pressure(state))
        if not np.isfinite(value):
            raise ValueError("boundary pressure must be finite")
        return float(np.clip(value, 0.0, 1.0))

    def encode(
        self,
        state: MotivationalState,
        profile: AppraisalProfile,
        features: StimulusFeatures,
        rule_state: MotivationalState,
    ) -> np.ndarray:
        self.manifest.validate_runtime(state, profile, features)
        if rule_state.schema != state.schema:
            raise ValueError("rule-appraised state schema changed unexpectedly")

        feature_values = np.asarray(
            [features.numeric(name, clamp=False) for name in self.manifest.feature_names],
            dtype=float,
        )
        vector = np.concatenate(
            [
                state.G.astype(float, copy=False),
                state.M.astype(float, copy=False),
                feature_values,
                rule_state.M.astype(float, copy=False),
                np.asarray([self.pressure(state)], dtype=float),
            ]
        )
        if vector.shape != (self.manifest.input_size,):
            raise ValueError("encoded learned-appraisal input has the wrong shape")
        if not np.all(np.isfinite(vector)):
            raise ValueError("encoded learned-appraisal input must be finite")

        if self.manifest.input_mean:
            means = np.asarray(self.manifest.input_mean, dtype=float)
            scales = np.asarray(self.manifest.input_scale, dtype=float)
            vector = (vector - means) / scales
        return vector
