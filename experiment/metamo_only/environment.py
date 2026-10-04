"""A small resource-choice environment with observable task/safety trade-offs.

MetaMo receives the same context but
selects actions only through OpenPsi appraisal and MAGUS decision scoring.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from core.features import StimulusFeatures
from core.schema import MagusGoalSchema, ModulatorSchema, MotivationSchema
from core.state import Action, MotivationalState
from magus.profile import DecisionProfile
from openpsi.profile import AppraisalProfile


SCHEMA = MotivationSchema(
    goals=MagusGoalSchema(
        primary_goals=("resource_gain", "energy_care", "information_gain"),
        anti_goals=("injury",),
    ),
    modulators=ModulatorSchema(),
    decision_profile_name="resource_choice_magus",
    appraisal_profile_name="resource_choice_openpsi",
)
FEATURES = ("opportunity", "hazard", "fatigue", "urgency", "uncertainty")


class ResourceAppraisalProfile(AppraisalProfile):
    """Conservative, imperfect rule appraisal used as the fixed baseline."""

    def core_modulator_deltas(self, state, features):
        opportunity = features.numeric("opportunity")
        hazard = features.numeric("hazard")
        fatigue = features.numeric("fatigue")
        urgency = features.numeric("urgency")
        uncertainty = features.numeric("uncertainty")
        return {
            "valence": 0.025 * (opportunity - hazard),
            "arousal": 0.025 * (hazard + urgency + uncertainty - 1.0),
            "approach": 0.035 * (opportunity + urgency - hazard - 0.5),
            "resolution": 0.025 * (opportunity + uncertainty - fatigue - 0.5),
            "threshold": 0.030 * (hazard + fatigue - 0.5),
            "securing": 0.030 * (hazard + fatigue - 0.5),
        }


APPRAISAL_PROFILE = ResourceAppraisalProfile(name=SCHEMA.appraisal_profile_name)
DECISION_PROFILE = DecisionProfile(
    name=SCHEMA.decision_profile_name,
    schema=SCHEMA,
    goal_modulators={
        "resource_gain": ("approach", "valence"),
        "energy_care": ("threshold", "securing"),
        "information_gain": ("resolution", "arousal"),
    },
    transcendence_goal_names=("resource_gain", "information_gain"),
    individuation_goal_names=("energy_care",),
    anti_goal_penalty_weights={"injury": 0.5},
    expected_goal_change_weight=0.0,
)


@dataclass(frozen=True)
class Case:
    state: MotivationalState
    stimulus: StimulusFeatures
    candidates: tuple[Action, ...]
    utilities: tuple[float, ...]
    rejected_by_hard_safety: int

    @property
    def oracle_index(self) -> int:
        return int(np.argmax(self.utilities))


def _action(name: str, correlations: dict[str, float], risk: float) -> Action:
    values = np.zeros(SCHEMA.num_goals, dtype=float)
    for goal, value in correlations.items():
        values[SCHEMA.goal_index(goal)] = float(value)
    return Action(
        id=name,
        goal_correlations=values,
        delta_g=np.zeros_like(values),
        schema=SCHEMA,
        metadata={"risk": float(risk), "opportunity": max(0.0, correlations.get("resource_gain", 0.0))},
    )


def sample_case(rng: np.random.Generator) -> Case:
    """Sample one independently seeded decision with three physical choices."""
    opportunity, hazard, fatigue, urgency, uncertainty = rng.uniform(0.05, 0.95, 5)
    context = dict(
        opportunity=float(opportunity), hazard=float(hazard),
        fatigue=float(fatigue), urgency=float(urgency),
        uncertainty=float(uncertainty),
    )
    stimulus = StimulusFeatures(context)
    goals = np.zeros(SCHEMA.num_goals)
    for name, value in {
        "individuation": 0.68,
        "transcendence": 0.55,
        "resource_gain": 0.68,
        "energy_care": 0.45,
        "information_gain": 0.48,
        "injury": 0.45,
    }.items():
        goals[SCHEMA.goal_index(name)] = value
    state = MotivationalState(goals, np.full(SCHEMA.num_modulators, 0.5), SCHEMA)

    gather_gain = 0.42 + 0.40 * opportunity
    gather_cost = 0.20 + 0.22 * fatigue
    gather_risk = 0.06 + 0.56 * hazard + 0.12 * uncertainty
    rest_gain = 0.40 + 0.42 * fatigue
    inspect_gain = 0.20 + 0.22 * opportunity
    inspect_info = 0.25 + 0.48 * uncertainty
    inspect_risk = 0.04 + 0.27 * hazard
    choices = (
        (
            _action("gather", {
                "resource_gain": gather_gain, "energy_care": -gather_cost,
                "information_gain": 0.0, "injury": gather_risk,
            }, gather_risk),
            0.95 * gather_gain * (0.65 + 0.55 * urgency)
            - 0.68 * gather_cost * fatigue
            - 1.1 * gather_risk * (0.35 + 0.55 * hazard),
        ),
        (
            _action("rest", {
                "resource_gain": 0.02, "energy_care": rest_gain,
                "information_gain": 0.0, "injury": 0.02,
            }, 0.02),
            0.68 * rest_gain * fatigue - 0.22 * urgency,
        ),
        (
            _action("inspect", {
                "resource_gain": inspect_gain, "energy_care": -0.10,
                "information_gain": inspect_info, "injury": inspect_risk,
            }, inspect_risk),
            0.48 * inspect_info * uncertainty * (1.0 - 0.45 * urgency)
            + 0.28 * inspect_gain - 0.70 * inspect_risk,
        ),
    )
    # This application-owned catastrophic-risk guard is identical in all arms.
    eligible = tuple((action, float(utility)) for action, utility in choices
                     if action.metadata["risk"] < 0.65)
    return Case(
        state=state,
        stimulus=stimulus,
        candidates=tuple(action for action, _ in eligible),
        utilities=tuple(utility for _, utility in eligible),
        rejected_by_hard_safety=len(choices) - len(eligible),
    )
