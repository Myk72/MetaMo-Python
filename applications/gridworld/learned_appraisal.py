"""Learned-appraisal wiring for the GridWorld application."""

from __future__ import annotations

from dataclasses import replace
import os

from category.bimonad import MetaMoPseudoBimonad
from dynamics.stability import boundary_pressure
from magus.decision import MagusDecision
from openpsi.appraisal import OpenPsiAppraisal
from openpsi.hybrid import (
    HybridAppraisalConfig,
    HybridAppraisalHistory,
    HybridAppraisalMode,
    HybridOpenPsiAppraisal,
)
from openpsi.learned import (
    LearnedAppraisalManifest,
    ManifestFeatureEncoder,
    ResidualPredictor,
    ZeroResidualPredictor,
)

from applications.gridworld.profile import (
    GRIDWORLD_APPRAISAL_PROFILE,
    GRIDWORLD_SCHEMA,
)
from applications.gridworld.runtime import GridWorldRuntime


GRIDWORLD_APPRAISAL_FEATURES = (
    "goal_proximity",
    "hazard_pressure",
    "energy_deficit",
    "time_pressure",
    "safe_progress",
    "safe_mobility",
)

GRIDWORLD_RESIDUAL_BOUNDS = (
    0.05,  # valence
    0.05,  # arousal
    0.05,  # approach
    0.05,  # resolution
    0.02,  # threshold
    0.02,  # securing
)


def gridworld_learned_appraisal_manifest() -> LearnedAppraisalManifest:
    """Return the frozen Phase-1 coordinate contract."""

    return LearnedAppraisalManifest(
        model_id="gridworld-zero-residual-v1",
        application_name="gridworld",
        appraisal_profile_name=GRIDWORLD_APPRAISAL_PROFILE.name,
        goal_names=tuple(GRIDWORLD_SCHEMA.goal_names),
        modulator_names=tuple(GRIDWORLD_SCHEMA.modulator_names),
        feature_names=GRIDWORLD_APPRAISAL_FEATURES,
        output_modulator_names=tuple(
            GRIDWORLD_SCHEMA.modulators.openpsi_modulators
        ),
        residual_bounds=GRIDWORLD_RESIDUAL_BOUNDS,
    )


def learned_appraisal_mode_from_environment() -> HybridAppraisalMode:
    """Read the opt-in runtime mode; disabled is the compatibility default."""

    raw = os.getenv("METAMO_LEARNED_APPRAISAL_MODE", "disabled")
    return HybridAppraisalMode(raw.strip().lower())


def make_gridworld_appraisal(
    mode: HybridAppraisalMode | str | None = None,
    predictor: ResidualPredictor | None = None,
    history: HybridAppraisalHistory | None = None,
    manifest: LearnedAppraisalManifest | None = None,
) -> HybridOpenPsiAppraisal:
    """Build the bounded wrapper; a predictor must be supplied for learned effects."""

    manifest = manifest or gridworld_learned_appraisal_manifest()
    selected_mode = (
        learned_appraisal_mode_from_environment()
        if mode is None
        else HybridAppraisalMode(mode)
    )
    return HybridOpenPsiAppraisal(
        base=OpenPsiAppraisal(profile=GRIDWORLD_APPRAISAL_PROFILE),
        manifest=manifest,
        encoder=ManifestFeatureEncoder(
            manifest=manifest,
            boundary_pressure=boundary_pressure,
        ),
        predictor=(
            predictor
            if predictor is not None
            else ZeroResidualPredictor(manifest.output_size)
        ),
        config=HybridAppraisalConfig(mode=selected_mode),
        history=history,
    )


def with_learned_appraisal(
    runtime: GridWorldRuntime,
    *,
    mode: HybridAppraisalMode | str,
    predictor: ResidualPredictor,
    manifest: LearnedAppraisalManifest,
    history: HybridAppraisalHistory | None = None,
) -> GridWorldRuntime:
    """Return an isolated opt-in runtime without altering GridWorld defaults."""

    appraisal = make_gridworld_appraisal(
        mode=mode, predictor=predictor, history=history, manifest=manifest,
    )
    return replace(
        runtime,
        name=f"{runtime.name}:hybrid-{HybridAppraisalMode(mode).value}",
        appraisal_profile=appraisal.profile,
        bimonad=MetaMoPseudoBimonad(
            appraisal,
            MagusDecision(profile=runtime.decision_profile),
        ),
    )
