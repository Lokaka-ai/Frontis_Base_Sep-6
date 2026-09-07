import json
import random
from pathlib import Path
import numpy as np
import pytest
from dojo.solvers.evo.evo import (
    Evolutionary,
    SolutionsDatabase,
    Island,
    capture_rng_state,
    restore_rng_state,
)
from dojo.core.solvers.utils.journal import Node
from dojo.core.solvers.utils.metric import MetricValue
from frontis_mila.recording import event
from frontis_mila.runtime import install


class Logger:
    def __getattr__(self, _):
        return lambda *a, **k: None


def database():
    db = SolutionsDatabase(
        num_islands=1,
        max_size=500,
        lower_is_better=True,
        logger=Logger(),
        experience_config={
            "enabled": True,
            "parent_selection": {
                "enabled": True,
                "selection_policy": "frontis",
                "weights": {"score": 1.0, "delta": 0.4, "novelty": 0.25},
            },
        },
    )
    nodes = [
        Node(
            id=str(i),
            code=f"x={i}",
            metric=MetricValue(0.1 + i * 0.03, maximize=False),
            is_buggy=False,
        )
        for i in range(4)
    ]
    db.seed_islands_with_nodes(nodes, [0] * 4)
    return db


def test_recording_preserves_rng_and_selection(tmp_path, monkeypatch):
    db = database()
    random.seed(101)
    np.random.seed(101)
    before = capture_rng_state()
    baseline = db.sample_in_context({"improve": 1, "crossover": 2}, 1.0, 0.5)
    after = capture_rng_state()
    classes = (Evolutionary, SolutionsDatabase, Island)
    saved = {c: dict(c.__dict__) for c in classes}
    monkeypatch.setenv("FRONTIS_RUN_DIR", str(tmp_path))
    try:
        install()
        restore_rng_state(before)
        instrumented = db.sample_in_context({"improve": 1, "crossover": 2}, 1.0, 0.5)
        assert ([n.id for n in baseline[0]], baseline[1:]) == (
            [n.id for n in instrumented[0]],
            instrumented[1:],
        )
        assert capture_rng_state() == after
        events = [
            json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()
        ]
        trace = events[-1]["trace"]
        assert sum(trace["operator_probabilities"].values()) == pytest.approx(1)
        assert sum(
            trace["legal_action_distribution"]["probabilities"]
        ) == pytest.approx(1)
    finally:
        for c, attrs in saved.items():
            for k in set(c.__dict__) - set(attrs):
                delattr(c, k)
            for k, v in attrs.items():
                if k not in ("__dict__", "__weakref__"):
                    setattr(c, k, v)


def test_observer_io_does_not_consume_rng(tmp_path, monkeypatch):
    monkeypatch.setenv("FRONTIS_RUN_DIR", str(tmp_path))
    before = capture_rng_state()
    event("test", value=float("inf"))
    assert capture_rng_state() == before
    assert json.loads((tmp_path / "events.jsonl").read_text())["value"] == "inf"


def test_factorized_audit_equals_full_distribution(monkeypatch, tmp_path):
    from dojo.solvers.evo.parent_policy import (
        build_action_audit,
        enumerate_legal_actions,
    )

    monkeypatch.setenv("FRONTIS_RUN_DIR", str(tmp_path))
    ids = ["a", "b", "c"]
    probabilities = [0.55, 0.3, 0.15]
    for operator in ("improve", "crossover"):
        for action in enumerate_legal_actions(operator, ids, probabilities):
            compact = build_action_audit(
                operator=operator,
                node_ids=ids,
                probabilities=probabilities,
                selected_node_ids=action["parent_node_ids"],
            )
            assert compact["selected_action_probability"] == pytest.approx(
                action["probability"]
            )
            assert compact["selected_action_id"] == action["action_id"]


def test_budget_stops_before_creating_guard_or_sampling(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from frontis_mila.runtime import BudgetComplete

    saved = {c: dict(c.__dict__) for c in (Evolutionary, SolutionsDatabase, Island)}
    monkeypatch.setenv("FRONTIS_RUN_DIR", str(tmp_path))
    try:
        install()
        solver = Evolutionary.__new__(Evolutionary)
        solver.cfg = SimpleNamespace(time_limit_secs=43200)
        solver.state = SimpleNamespace(running_time=43200)
        before = capture_rng_state()
        with pytest.raises(BudgetComplete, match="budget_exhausted"):
            solver._begin_candidate_slot(7, 2)
        assert capture_rng_state() == before
        assert not list(tmp_path.rglob("inflight_slot.json"))
    finally:
        for c, attrs in saved.items():
            for k in set(c.__dict__) - set(attrs):
                delattr(c, k)
            for k, v in attrs.items():
                if k not in ("__dict__", "__weakref__"):
                    setattr(c, k, v)
