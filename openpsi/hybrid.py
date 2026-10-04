"""Safe composition of rule-based and optional learned appraisal."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np

from category.functors import AppraisalComonad
from core.features import GoalChangeFeedback, StimulusFeatures
from core.state import MotivationalState
from openpsi.appraisal import OpenPsiAppraisal
from openpsi.learned import (
    LearnedAppraisalManifest,
    ManifestFeatureEncoder,
    ResidualPredictor,
)


class HybridAppraisalMode(str, Enum):
    """Runtime participation level for learned appraisal."""

    DISABLED = "disabled"
    SHADOW = "shadow"
    ENABLED = "enabled"


@dataclass(frozen=True)
class HybridAppraisalConfig:
    """Runtime safety controls independent of learned weights."""

    mode: HybridAppraisalMode = HybridAppraisalMode.DISABLED
    nonnegative_modulators: tuple[str, ...] = ("threshold", "securing")
    residual_norm_cap: float = 0.08
    attenuate_non_safety_near_boundary: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            object.__setattr__(self, "mode", HybridAppraisalMode(self.mode))
        if not np.isfinite(self.residual_norm_cap) or self.residual_norm_cap < 0.0:
            raise ValueError("residual_norm_cap must be finite and non-negative")
        if len(set(self.nonnegative_modulators)) != len(
            self.nonnegative_modulators
        ):
            raise ValueError("nonnegative_modulators must contain unique names")


@dataclass(frozen=True)
class HybridAppraisalRecord:
    """One rule, shadow, applied, or fail-closed appraisal observation."""

    mode: str
    status: str
    model_id: str
    profile_name: str
    rule_modulators: tuple[float, ...]
    predicted_normalized: tuple[float, ...]
    proposed_residual: tuple[float, ...]
    applied_residual: tuple[float, ...]
    result_modulators: tuple[float, ...]
    boundary_pressure: float
    prediction_clipped: bool = False
    fallback_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class HybridAppraisalHistory:
    """In-memory Phase-1 telemetry collector."""

    def __init__(self, max_records: int | None = 10000) -> None:
        if max_records is not None and max_records <= 0:
            raise ValueError("max_records must be positive or None")
        self.max_records = max_records
        self.records: list[HybridAppraisalRecord] = []
        self.dropped_records = 0

    def append(self, record: HybridAppraisalRecord) -> None:
        self.records.append(record)
        if self.max_records is not None and len(self.records) > self.max_records:
            excess = len(self.records) - self.max_records
            del self.records[:excess]
            self.dropped_records += excess

    def clear(self) -> None:
        self.records.clear()
        self.dropped_records = 0

    def last(self) -> HybridAppraisalRecord | None:
        if not self.records:
            return None
        return self.records[-1]

    def to_rows(self) -> list[dict]:
        return [record.to_dict() for record in self.records]

    def __len__(self) -> int:
        return len(self.records)


def _project_appraisal_residual(
    manifest: LearnedAppraisalManifest,
    config: HybridAppraisalConfig,
    normalized_prediction: np.ndarray,
    boundary_pressure: float,
) -> tuple[np.ndarray, bool, dict[str, Any]]:
    """Project one prediction and expose audit-only constraint activity."""

    prediction = np.asarray(normalized_prediction, dtype=float)
    if prediction.shape != (manifest.output_size,):
        raise ValueError(
            "learned residual prediction shape does not match manifest outputs"
        )
    if not np.all(np.isfinite(prediction)):
        raise ValueError("learned residual prediction must be finite")
    if not np.isfinite(boundary_pressure):
        raise ValueError("boundary pressure must be finite")
    pressure = float(np.clip(boundary_pressure, 0.0, 1.0))

    clipped = bool(np.any((prediction < -1.0) | (prediction > 1.0)))
    prediction = np.clip(prediction, -1.0, 1.0)
    bounds = np.asarray(manifest.residual_bounds, dtype=float)
    residual = prediction * bounds

    nonnegative = set(config.nonnegative_modulators)
    direction_clipped: list[str] = []
    boundary_attenuated: list[str] = []
    coordinate_saturated: list[str] = []
    for idx, name in enumerate(manifest.output_modulator_names):
        if bounds[idx] > 0.0 and abs(prediction[idx]) >= 1.0 - 1e-12:
            coordinate_saturated.append(name)
        if name in nonnegative:
            if residual[idx] < 0.0:
                direction_clipped.append(name)
            residual[idx] = max(0.0, residual[idx])
        elif config.attenuate_non_safety_near_boundary:
            if pressure > 0.0 and residual[idx] != 0.0:
                boundary_attenuated.append(name)
            residual[idx] *= 1.0 - pressure

    pre_norm = float(np.linalg.norm(residual))
    norm_capped = pre_norm > config.residual_norm_cap and pre_norm > 0.0
    if norm_capped:
        residual *= config.residual_norm_cap / pre_norm
    diagnostics = {
        "prediction_clipped": clipped,
        "coordinate_saturated": coordinate_saturated,
        "direction_clipped": direction_clipped,
        "boundary_attenuated": boundary_attenuated,
        "norm_capped": norm_capped,
        "pre_norm": pre_norm,
        "projected_norm": float(np.linalg.norm(residual)),
    }
    return residual, clipped, diagnostics


def bound_appraisal_residual(
    manifest: LearnedAppraisalManifest,
    config: HybridAppraisalConfig,
    normalized_prediction: np.ndarray,
    boundary_pressure: float,
) -> tuple[np.ndarray, bool]:
    """Apply the exact runtime direction, coordinate, boundary, and norm caps.

    Offline target search uses this same function so a generated target cannot
    request a residual that the runtime wrapper would later alter.
    """

    residual, clipped, _ = _project_appraisal_residual(
        manifest,
        config,
        normalized_prediction,
        boundary_pressure,
    )
    return residual, clipped


def appraisal_residual_constraint_activity(
    manifest: LearnedAppraisalManifest,
    config: HybridAppraisalConfig,
    normalized_prediction: np.ndarray,
    boundary_pressure: float,
) -> dict[str, Any]:
    """Describe projection constraints without changing runtime behavior."""

    _, _, diagnostics = _project_appraisal_residual(
        manifest,
        config,
        normalized_prediction,
        boundary_pressure,
    )
    return diagnostics


class HybridOpenPsiAppraisal(AppraisalComonad):
    """Decorate an existing OpenPsi appraiser with a bounded residual path.

    Disabled mode returns the rule result directly.  Shadow mode executes and
    records the residual path but applies a zero residual.  Enabled mode is
    provided for guarded tests and later phases; Phase 1 wires a zero predictor
    by default, so all three modes preserve rule behavior.
    """

    def __init__(
        self,
        base: OpenPsiAppraisal,
        manifest: LearnedAppraisalManifest,
        encoder: ManifestFeatureEncoder,
        predictor: ResidualPredictor,
        config: HybridAppraisalConfig | None = None,
        history: HybridAppraisalHistory | None = None,
    ) -> None:
        if encoder.manifest != manifest:
            raise ValueError("feature encoder and hybrid appraiser manifests differ")
        self.base = base
        self.profile = base.profile
        self.manifest = manifest
        self.encoder = encoder
        self.predictor = predictor
        self.config = config or HybridAppraisalConfig()
        self.history = history if history is not None else HybridAppraisalHistory()
        self.manifest.validate_artifact()

    def extract(self, state: MotivationalState) -> MotivationalState:
        return self.base.extract(state)

    def appraise(self, state: MotivationalState, stimulus: Any) -> MotivationalState:
        features = self.profile.stimulus_features(state, stimulus)
        rule_state = self.profile.appraise_features(state, features)
        return self._apply_or_observe(state, features, rule_state)

    def goal_change_feedback(
        self,
        state: MotivationalState,
        stimulus: Any,
    ) -> GoalChangeFeedback:
        return self.base.goal_change_feedback(state, stimulus)

    def appraise_with_feedback(
        self,
        state: MotivationalState,
        stimulus: Any,
    ) -> tuple[MotivationalState, GoalChangeFeedback]:
        """Extract features once and preserve rule-owned goal feedback."""

        features = self.profile.stimulus_features(state, stimulus)
        rule_state = self.profile.appraise_features(state, features)
        feedback = self.profile.goal_change_feedback(state, stimulus, features)
        result_state = self._apply_or_observe(state, features, rule_state)
        return result_state, feedback

    def _record_fallback(
        self,
        rule_state: MotivationalState,
        error: Exception,
    ) -> None:
        reason = f"{type(error).__name__}: {error}"
        self.history.append(
            HybridAppraisalRecord(
                mode=self.config.mode.value,
                status="fallback",
                model_id=self.manifest.model_id,
                profile_name=self.profile.name,
                rule_modulators=tuple(float(value) for value in rule_state.M),
                predicted_normalized=(),
                proposed_residual=(),
                applied_residual=tuple(0.0 for _ in rule_state.M),
                result_modulators=tuple(float(value) for value in rule_state.M),
                boundary_pressure=0.0,
                fallback_reason=reason,
            )
        )

    def _bounded_residual(
        self,
        normalized_prediction: np.ndarray,
        boundary_pressure: float,
    ) -> tuple[np.ndarray, bool]:
        return bound_appraisal_residual(
            self.manifest,
            self.config,
            normalized_prediction,
            boundary_pressure,
        )

    def _apply_or_observe(
        self,
        state: MotivationalState,
        features: StimulusFeatures,
        rule_state: MotivationalState,
    ) -> MotivationalState:
        if self.config.mode is HybridAppraisalMode.DISABLED:
            return rule_state

        try:
            encoded = self.encoder.encode(
                state,
                self.profile,
                features,
                rule_state,
            )
            prediction = np.asarray(self.predictor.predict(encoded), dtype=float)
            pressure = self.encoder.pressure(state)
            residual, prediction_clipped = self._bounded_residual(
                prediction,
                pressure,
            )

            result = rule_state.copy()
            applied = np.zeros_like(result.M)
            if self.config.mode is HybridAppraisalMode.ENABLED:
                for output_idx, name in enumerate(
                    self.manifest.output_modulator_names
                ):
                    state_idx = result.modulator_index(name)
                    applied[state_idx] = residual[output_idx]
                result.M = np.clip(result.M + applied, 0.0, 1.0)
                applied = result.M - rule_state.M
                status = "applied"
            else:
                status = "shadow"

            self.history.append(
                HybridAppraisalRecord(
                    mode=self.config.mode.value,
                    status=status,
                    model_id=self.manifest.model_id,
                    profile_name=self.profile.name,
                    rule_modulators=tuple(float(value) for value in rule_state.M),
                    predicted_normalized=tuple(float(value) for value in prediction),
                    proposed_residual=tuple(float(value) for value in residual),
                    applied_residual=tuple(float(value) for value in applied),
                    result_modulators=tuple(float(value) for value in result.M),
                    boundary_pressure=pressure,
                    prediction_clipped=prediction_clipped,
                )
            )
            return result
        except Exception as error:
            self._record_fallback(rule_state, error)
            return rule_state
