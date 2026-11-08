"""Immutable observations and frozen forecasts for GridWorld transitions.

Records and forecasts are diagnostic data. They do not participate in action
selection or change the committed motivational state.
"""

from dataclasses import dataclass
from math import isfinite

from applications.gridworld.stimulus import GridWorldStimulus
from core.state import MotivationalState

# A snapshot of the motivational state at a given moment in time.
@dataclass(frozen=True)
class MotivationalSnapshot:
    goal_names: tuple[str, ...]
    goals: tuple[float, ...]
    modulator_names: tuple[str, ...]
    modulators: tuple[float, ...]

    @classmethod
    def from_state(cls, state: MotivationalState) -> "MotivationalSnapshot":
        return cls(
            goal_names=state.schema.goal_names,
            goals=tuple(float(value) for value in state.G),
            modulator_names=state.schema.modulator_names,
            modulators=tuple(float(value) for value in state.M),
        )


@dataclass(frozen=True)
class MoveForecastInputs:
    """Selected candidate estimates available before the environment moves."""

    hazard_risk: float
    mineral_progress: float
    mineral_collected: float
    safe_mobility_after: float

    def __post_init__(self):
        if not all(
            isfinite(value) and 0.0 <= value <= 1.0
            for value in (
                self.hazard_risk,
                self.mineral_progress,
                self.mineral_collected,
                self.safe_mobility_after,
            )
        ):
            raise ValueError("move forecast inputs must be finite and within [0, 1]")

# A pending transition is one that has been initiated but not yet completed.
@dataclass(frozen=True)
class GridWorldSelfForecast:
    transition_id: int
    action_id: str
    predicted_committed_state: MotivationalSnapshot
    predicted_next_stimulus: GridWorldStimulus
    predicted_next_appraisal: MotivationalSnapshot


@dataclass(frozen=True)
class PendingGridWorldTransition:
    transition_id: int
    episode_id: int
    step_index: int
    action_index: int
    action_id: str
    current_state: MotivationalSnapshot
    current_stimulus: GridWorldStimulus
    staged_state: MotivationalSnapshot
    move_inputs: MoveForecastInputs | None = None
    forecast: GridWorldSelfForecast | None = None

# A committed transition is one that has been completed and recorded.
@dataclass(frozen=True)
class GridWorldTransitionRecord:
    transition_id: int
    episode_id: int
    step_index: int
    action_index: int
    action_id: str
    current_state: MotivationalSnapshot
    current_stimulus: GridWorldStimulus
    staged_state: MotivationalSnapshot
    committed_state: MotivationalSnapshot
    next_stimulus: GridWorldStimulus
    done: bool
    move_inputs: MoveForecastInputs | None = None
    forecast: GridWorldSelfForecast | None = None
    observed_next_appraisal: MotivationalSnapshot | None = None
