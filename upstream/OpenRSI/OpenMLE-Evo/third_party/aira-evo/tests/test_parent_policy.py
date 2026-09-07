from __future__ import annotations

import numpy as np
import pytest

from dojo.solvers.evo.parent_policy import (
    build_action_audit,
    enumerate_legal_actions,
    normalize_probabilities,
    sample_without_replacement,
)


def test_improve_actions_match_node_probabilities():
    actions = enumerate_legal_actions("improve", ["a", "b"], [0.25, 0.75])
    assert actions == [
        {"action_id": "improve:a", "parent_node_ids": ["a"], "probability": 0.25},
        {"action_id": "improve:b", "parent_node_ids": ["b"], "probability": 0.75},
    ]


def test_crossover_actions_are_ordered_and_have_exact_sequential_mass():
    actions = enumerate_legal_actions("crossover", ["a", "b", "c"], [0.5, 0.3, 0.2])
    by_id = {action["action_id"]: action["probability"] for action in actions}
    assert len(actions) == 6
    assert by_id["crossover:a->b"] == pytest.approx(0.5 * 0.3 / 0.5)
    assert by_id["crossover:b->a"] == pytest.approx(0.3 * 0.5 / 0.7)
    assert by_id["crossover:a->b"] != by_id["crossover:b->a"]
    assert sum(by_id.values()) == pytest.approx(1.0)


def test_sequential_sampler_never_repeats_a_parent():
    rng = np.random.RandomState(7)
    for _ in range(100):
        selected = sample_without_replacement([0.8, 0.15, 0.05], 2, rng=rng)
        assert len(selected) == len(set(selected)) == 2


def test_action_audit_identifies_the_ordered_selected_pair():
    audit = build_action_audit(
        operator="crossover",
        node_ids=["a", "b", "c"],
        probabilities=[0.5, 0.3, 0.2],
        selected_node_ids=["b", "a"],
    )
    assert audit["action_semantics"] == "ordered_parent_pair"
    assert audit["selected_action_id"] == "crossover:b->a"


def test_invalid_or_degenerate_weights_are_handled_explicitly():
    assert normalize_probabilities([0.0, 0.0]) == [0.5, 0.5]
    with pytest.raises(ValueError):
        normalize_probabilities([1.0, -1.0])
    with pytest.raises(ValueError):
        enumerate_legal_actions("crossover", ["a"], [1.0])
