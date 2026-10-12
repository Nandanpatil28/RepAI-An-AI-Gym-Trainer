"""Focused counter tests that do not require a camera or MediaPipe inference."""

from exercise_engine import WorkoutEngine


def bare_engine():
    engine = object.__new__(WorkoutEngine)
    engine.counters = {exercise: 0 for exercise in ("push-up", "pull-up", "sit-up", "squat", "walk")}
    engine.statuses = {exercise: True for exercise in engine.counters}
    engine.rep_states = {exercise: engine._new_rep_state() for exercise in engine.counters}
    return engine


def walking_metrics(left_ankle_y, right_ankle_y):
    return {
        "torso_y_delta": 0.20,
        "leg": 165,
        "left_ankle_y": left_ankle_y,
        "right_ankle_y": right_ankle_y,
    }


def test_walking_counts_only_confirmed_alternating_steps():
    engine = bare_engine()

    # Establish a neutral stance, then hold a raised left foot long enough to
    # count once. Holding it still must not add duplicate steps.
    engine._count_walk_step("walk", walking_metrics(0.90, 0.90))
    engine._count_walk_step("walk", walking_metrics(0.82, 0.90))
    assert engine._count_walk_step("walk", walking_metrics(0.82, 0.90))
    for _ in range(4):
        assert not engine._count_walk_step("walk", walking_metrics(0.82, 0.90))
    assert engine.counters["walk"] == 1

    # Return to neutral, then lift the other foot. This is the second step.
    engine._count_walk_step("walk", walking_metrics(0.90, 0.90))
    engine._count_walk_step("walk", walking_metrics(0.90, 0.82))
    assert engine._count_walk_step("walk", walking_metrics(0.90, 0.82))
    assert engine.counters["walk"] == 2


def test_rep_phase_counter_is_unchanged_for_other_exercises():
    engine = bare_engine()
    assert not engine._update_phase_counter("squat", True, False)
    assert not engine._update_phase_counter("squat", True, False)
    assert not engine._update_phase_counter("squat", False, True)
    assert not engine._update_phase_counter("squat", False, True)
    assert not engine._update_phase_counter("squat", True, False)
    assert engine._update_phase_counter("squat", True, False)
    assert engine.counters["squat"] == 1
