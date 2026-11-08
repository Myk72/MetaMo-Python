"""An observation-only GridWorld self-model with online stimulus learning."""

from collections import deque
from dataclasses import astuple, dataclass
from math import ceil, floor, fsum, isfinite
from statistics import median

from applications.gridworld.config import MAX_MANHATTAN_DISTANCE, MAX_STEPS, REWARD_LAVA
from applications.gridworld.stimulus import GridWorldStimulus
from applications.gridworld.transition_record import (
    GridWorldSelfForecast,
    GridWorldTransitionRecord,
    MotivationalSnapshot,
    MoveForecastInputs,
)
from core.state import MotivationalState
from openpsi.profile import AppraisalProfile

# mae = mean absolute error
def _mae(expected: tuple[float, ...], actual: tuple[float, ...]) -> float:
    if len(expected) != len(actual):
        raise ValueError("forecast and observation have different dimensions")
    return fsum(abs(a - b) for a, b in zip(expected, actual)) / len(expected)


def _features(stimulus: GridWorldStimulus) -> tuple[float, ...]:
    values = tuple(float(value) for value in astuple(stimulus))
    if not all(isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise ValueError("GridWorld stimulus features must be finite and within [0, 1]")
    return values


@dataclass(frozen=True)
class ActionFeatureStats:
    count: int
    mean_change: tuple[float, ...]


@dataclass(frozen=True)
class GridWorldForecastError:
    transition_id: int
    committed_goal_mae: float
    committed_modulator_mae: float
    next_stimulus_mae: float
    next_appraisal_modulator_mae: float | None


class GridWorldPredictiveSelfModel:
    """Forecast without changing live MetaMo state or action selection."""

    def __init__(
        self,
        appraisal_profile: AppraisalProfile,
        *,
        min_samples: int = 10,
        prior_count: float = 20.0,
        min_collection_samples: int = 5,
        collection_history_limit: int = 256,
        forecast_mode: str = "move_informed",
    ):
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1")
        if not isfinite(prior_count) or prior_count < 0.0:
            raise ValueError("prior_count must be finite and non-negative")
        if min_collection_samples < 1 or collection_history_limit < min_collection_samples:
            raise ValueError("collection history must fit the minimum sample count")
        if forecast_mode not in {"move_informed", "move_nominal", "contextual", "action_only"}:
            raise ValueError("unknown forecast_mode")
        self.appraisal_profile = appraisal_profile
        self.min_samples = min_samples
        self.prior_count = prior_count
        self.min_collection_samples = min_collection_samples
        self.forecast_mode = forecast_mode
        self.learning_enabled = True
        self._stats: dict[str, ActionFeatureStats] = {}
        self._context_stats: dict[tuple[str, bool, bool], ActionFeatureStats] = {}
        self._collection_proximities: deque[float] = deque(maxlen=collection_history_limit)
        self._collection_total_count = 0
        self._observed_ids: set[int] = set()

    @property
    def collection_sample_count(self) -> int:
        return len(self._collection_proximities)

    @property
    def collection_observation_count(self) -> int:
        return self._collection_total_count

    def collection_proximity_interval(self) -> tuple[float, float] | None:
        """Empirical 10th to 90th percentile for the post-respawn signal."""
        if len(self._collection_proximities) < self.min_collection_samples:
            return None
        values = sorted(self._collection_proximities)
        last = len(values) - 1
        return values[floor(0.1 * last)], values[ceil(0.9 * last)]

    @staticmethod
    def _context_key(action_id: str, stimulus: GridWorldStimulus) -> tuple[str, bool, bool]:
        """Use only pre-action signals; thresholds are fixed before evaluation."""
        return (action_id, stimulus.hazard_pressure >= 0.5, stimulus.safe_mobility < 0.5)

    def context_sample_count(self, action_id: str, stimulus: GridWorldStimulus) -> int:
        stats = self._context_stats.get(self._context_key(action_id, stimulus))
        return 0 if stats is None else stats.count

    def sample_count(self, action_id: str) -> int:
        stats = self._stats.get(action_id)
        return 0 if stats is None else stats.count

    def mean_change(self, action_id: str) -> tuple[float, ...] | None:
        stats = self._stats.get(action_id)
        return None if stats is None else stats.mean_change

    def forecast(
        self,
        *,
        transition_id: int,
        action_id: str,
        staged_state: MotivationalState,
        current_stimulus: GridWorldStimulus,
        move_inputs: MoveForecastInputs | None = None,
    ) -> GridWorldSelfForecast:
        if not action_id:
            raise ValueError("action_id must be non-empty")
        current_features = _features(current_stimulus)
        stats = (
            self._context_stats.get(self._context_key(action_id, current_stimulus))
            if self.forecast_mode in {"move_informed", "contextual"}
            else self._stats.get(action_id)
        )
        if self.forecast_mode == "move_nominal" or stats is None or stats.count < self.min_samples:
            predicted_stimulus = current_stimulus
        else:
            weight = stats.count / (stats.count + self.prior_count)
            predicted_stimulus = GridWorldStimulus(
                *(
                    max(0.0, min(1.0, value + weight * change))
                    for value, change in zip(current_features, stats.mean_change)
                )
            )
        if self.forecast_mode in {"move_informed", "move_nominal"} and move_inputs is not None:
            # Candidate values were computed from the current environment state.
            # The new mineral location after collection is random. Use the
            # collection-only median when enough outcomes have been observed.
            if move_inputs.mineral_collected >= 0.5:
                if (
                    self.forecast_mode == "move_informed"
                    and self.collection_sample_count >= self.min_collection_samples
                ):
                    proximity = float(median(self._collection_proximities))
                else:
                    proximity = current_stimulus.goal_proximity
            else:
                proximity = max(
                    0.0,
                    min(
                        1.0,
                        current_stimulus.goal_proximity
                        + (2.0 * move_inputs.mineral_progress - 1.0)
                        / MAX_MANHATTAN_DISTANCE,
                    ),
                )
            energy_deficit = (
                0.0
                if move_inputs.mineral_collected >= 0.5
                else max(0.0, min(1.0, current_stimulus.energy_deficit
                                   - (REWARD_LAVA / 100.0 if move_inputs.hazard_risk == 1.0 else 0.0)))
            )
            predicted_stimulus = GridWorldStimulus(
                goal_proximity=proximity,
                hazard_pressure=move_inputs.hazard_risk,
                energy_deficit=energy_deficit,
                time_pressure=min(1.0, current_stimulus.time_pressure + 1.0 / MAX_STEPS),
                safe_progress=proximity * (1.0 - move_inputs.hazard_risk),
                safe_mobility=move_inputs.safe_mobility_after,
            )
        predicted_appraisal = self.appraisal_profile.appraise(
            staged_state, predicted_stimulus
        )
        return GridWorldSelfForecast(
            transition_id=transition_id,
            action_id=action_id,
            predicted_committed_state=MotivationalSnapshot.from_state(staged_state),
            predicted_next_stimulus=predicted_stimulus,
            predicted_next_appraisal=MotivationalSnapshot.from_state(
                predicted_appraisal
            ),
        )

    def observe(self, record: GridWorldTransitionRecord) -> None:
        """Update once, after a real nonterminal transition is committed."""
        forecast = record.forecast
        if (
            forecast is None
            or forecast.transition_id != record.transition_id
            or forecast.action_id != record.action_id
        ):
            raise ValueError("record has no matching frozen forecast")
        if record.transition_id in self._observed_ids:
            raise ValueError("transition has already been observed")
        if record.done or not self.learning_enabled:
            self._observed_ids.add(record.transition_id)
            return

        current = _features(record.current_stimulus)
        following = _features(record.next_stimulus)
        change = tuple(after - before for before, after in zip(current, following))
        for table, key in (
            (self._stats, record.action_id),
            (self._context_stats, self._context_key(record.action_id, record.current_stimulus)),
        ):
            old = table.get(key, ActionFeatureStats(count=0, mean_change=(0.0,) * len(change)))
            count = old.count + 1
            mean = tuple(
                value + (delta - value) / count
                for value, delta in zip(old.mean_change, change)
            )
            table[key] = ActionFeatureStats(count=count, mean_change=mean)
        if record.move_inputs is not None and record.move_inputs.mineral_collected >= 0.5:
            self._collection_proximities.append(record.next_stimulus.goal_proximity)
            self._collection_total_count += 1
        self._observed_ids.add(record.transition_id)

    @staticmethod
    def evaluate(record: GridWorldTransitionRecord) -> GridWorldForecastError:
        """Compare a frozen forecast with its paired real transition."""
        forecast = record.forecast
        if (
            forecast is None
            or forecast.transition_id != record.transition_id
            or forecast.action_id != record.action_id
        ):
            raise ValueError("record has no matching frozen forecast")
        if forecast.predicted_committed_state.goal_names != record.committed_state.goal_names:
            raise ValueError("forecast and committed goal schemas differ")
        if (
            forecast.predicted_committed_state.modulator_names
            != record.committed_state.modulator_names
        ):
            raise ValueError("forecast and committed modulator schemas differ")

        appraisal_error = None
        if record.observed_next_appraisal is not None:
            if (
                forecast.predicted_next_appraisal.modulator_names
                != record.observed_next_appraisal.modulator_names
            ):
                raise ValueError("forecast and observed appraisal schemas differ")
            appraisal_error = _mae(
                forecast.predicted_next_appraisal.modulators,
                record.observed_next_appraisal.modulators,
            )
        return GridWorldForecastError(
            transition_id=record.transition_id,
            committed_goal_mae=_mae(
                forecast.predicted_committed_state.goals,
                record.committed_state.goals,
            ),
            committed_modulator_mae=_mae(
                forecast.predicted_committed_state.modulators,
                record.committed_state.modulators,
            ),
            next_stimulus_mae=_mae(
                astuple(forecast.predicted_next_stimulus),
                astuple(record.next_stimulus),
            ),
            next_appraisal_modulator_mae=appraisal_error,
        )
