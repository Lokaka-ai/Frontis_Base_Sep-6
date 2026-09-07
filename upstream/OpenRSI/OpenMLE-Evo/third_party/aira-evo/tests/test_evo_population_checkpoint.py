import random
from types import SimpleNamespace

import numpy
import pytest

from dojo.core.solvers.llm_helpers.generic_llm import GenericLLM
from dojo.core.solvers.operators.core import CodeExtractionError
from dojo.core.solvers.utils.journal import Journal, Node
from dojo.core.solvers.utils.metric import MetricValue, WorstMetricValue
from dojo.solvers.evo.evo import (
    Evolutionary,
    SolutionsDatabase,
    capture_rng_state,
    restore_rng_state,
)
from dojo.utils.state import EvolutionaryState


class NullLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def make_node(node_id: str, value: float) -> Node:
    return Node(
        code=f"# {node_id}",
        id=node_id,
        metric=MetricValue(value, maximize=False),
        is_buggy=False,
    )


def test_evolutionary_state_persists_generation():
    state = EvolutionaryState()
    state.current_step = 7
    state.current_generation = 3
    state.running_time = 12.5
    payload = state.state_dict()

    restored = EvolutionaryState()
    restored.load_state_dict(payload)

    assert restored.current_step == 7
    assert restored.current_generation == 3
    assert restored.running_time == 12.5


def test_debug_extraction_failure_is_recorded_and_ends_cycle():
    solver = Evolutionary.__new__(Evolutionary)
    solver.cfg = SimpleNamespace(
        max_debug_depth=1,
        max_debug_time=60,
        experience={"enabled": False},
    )
    solver.lower_is_better = True
    solver.logger = NullLogger()
    parent = Node(
        code="broken",
        metric=WorstMetricValue(maximize=False),
        is_buggy=True,
        exec_time=0,
    )
    trace = {"completion_text": "analysis without code"}

    def fail_debug(_node):
        raise CodeExtractionError("no Python", attempt_metrics=[trace])

    solver._debug = fail_debug
    state, path, fixed_metric = solver.debug_cycle(object(), None, parent)

    assert state is not None
    assert fixed_metric is None
    assert len(path) == 2
    failure = path[-1]
    assert failure.parents == [parent]
    assert failure.operators_metrics == [trace]
    assert failure.metric.info["failure_type"] == "code_extraction"


def test_population_state_round_trip_preserves_exact_membership_and_order():
    nodes = {
        "a": make_node("a", 0.2),
        "b": make_node("b", 0.3),
        "c": make_node("c", 0.4),
    }
    source = SolutionsDatabase(
        num_islands=2,
        max_size=10,
        lower_is_better=True,
        logger=NullLogger(),
    )
    source._islands[0].nodes = [nodes["b"], nodes["a"]]
    source._islands[1].nodes = [nodes["c"]]
    source.global_min_fitness = 0.2
    source.global_max_fitness = 0.4
    source.request_fresh_draft("test_reason")

    payload = source.state_dict()
    restored = SolutionsDatabase(
        num_islands=2,
        max_size=10,
        lower_is_better=True,
        logger=NullLogger(),
    )
    restored.load_state_dict(payload, nodes_by_id=nodes)

    assert [[node.id for node in island.nodes] for island in restored._islands] == [
        ["b", "a"],
        ["c"],
    ]
    assert restored.global_min_fitness == 0.2
    assert restored.global_max_fitness == 0.4
    assert restored._forced_fresh_draft_reason == "test_reason"


def test_persisted_forced_improve_selects_exact_live_parent():
    nodes = [make_node("a", 0.2), make_node("b", 0.3), make_node("c", 0.4)]
    database = SolutionsDatabase(
        num_islands=1,
        max_size=10,
        lower_is_better=True,
        logger=NullLogger(),
        experience_config={
            "enabled": False,
            "parent_selection": {
                "forced_action": {
                    "operator": "improve",
                    "parent_node_ids": ["b"],
                }
            },
        },
    )
    database._islands[0].nodes = nodes

    selected, island_id, operator = database.sample_in_context(
        {"improve": 1, "crossover": 2},
        temperature=1.0,
        crossover_prob=1.0,
        decision_state_id="engineering-state",
    )

    assert operator == "improve"
    assert island_id == 0
    assert [node.id for node in selected] == ["b"]
    assert database.last_parent_selection["forced_action"] == {
        "operator": "improve",
        "parent_node_ids": ["b"],
    }
    assert database.last_parent_selection["selected_action_id"] == "improve:b"


def test_rng_state_round_trip_reproduces_python_and_numpy_draws():
    random.seed(12345)
    numpy.random.seed(54321)
    payload = capture_rng_state()
    expected_python = [random.random() for _ in range(4)]
    expected_numpy = numpy.random.random(4)

    random.seed(1)
    numpy.random.seed(1)
    restore_rng_state(payload)

    assert [random.random() for _ in range(4)] == expected_python
    numpy.testing.assert_array_equal(numpy.random.random(4), expected_numpy)


def test_cloned_population_reproduces_candidate_vector_and_action_draw():
    nodes = {
        "a": make_node("a", 0.2),
        "b": make_node("b", 0.3),
        "c": make_node("c", 0.4),
    }
    experience = {
        "enabled": True,
        "parent_selection": {
            "enabled": True,
            "weights": {"score": 1.0, "delta": 0.4, "novelty": 0.25},
        },
    }
    for node in nodes.values():
        node.experience_card = {
            "node_id": node.id,
            "method_family_auto": "sklearn",
            "fitness": node.metric.value,
            "status": "success",
            "status_code": 200,
            "is_buggy": False,
        }
    source = SolutionsDatabase(
        num_islands=1,
        max_size=10,
        lower_is_better=True,
        logger=NullLogger(),
        experience_config=experience,
    )
    source._islands[0].nodes = [nodes["b"], nodes["a"], nodes["c"]]
    source.global_min_fitness = 0.2
    source.global_max_fitness = 0.4

    random.seed(2026)
    numpy.random.seed(2027)
    population_payload = source.state_dict()
    rng_payload = capture_rng_state()
    expected_nodes, expected_island, expected_operator = source.sample_in_context(
        {"improve": 1, "crossover": 2}, 1.0, 0.5
    )
    expected_trace = source.last_parent_selection

    clone = SolutionsDatabase(
        num_islands=1,
        max_size=10,
        lower_is_better=True,
        logger=NullLogger(),
        experience_config=experience,
    )
    clone.load_state_dict(population_payload, nodes_by_id=nodes)
    restore_rng_state(rng_payload)
    actual_nodes, actual_island, actual_operator = clone.sample_in_context(
        {"improve": 1, "crossover": 2}, 1.0, 0.5
    )

    assert [node.id for node in actual_nodes] == [node.id for node in expected_nodes]
    assert (actual_island, actual_operator) == (expected_island, expected_operator)
    assert clone.last_parent_selection == expected_trace


def test_llm_trace_records_exact_prompt_and_request_seed():
    llm = GenericLLM.__new__(GenericLLM)
    llm.generation_kwargs = {"temperature": 0.6}
    llm.call_tracker = 99
    llm.set_generation_seed(20260816)
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user prompt"},
    ]

    llm.call_tracker += 1
    first_kwargs = llm._generation_kwargs_for_call()
    llm.call_tracker += 1
    second_kwargs = llm._generation_kwargs_for_call()
    trace = llm._build_trace(
        usage_stats={"prompt_tokens": 4},
        messages=messages,
        output="completion",
        generation_kwargs=second_kwargs,
    )

    assert trace["prompt_messages"] == messages
    assert trace["completion_text"] == "completion"
    assert first_kwargs["seed"] == 20260816
    assert trace["generation_kwargs"]["seed"] == 20260817

    clone = GenericLLM.__new__(GenericLLM)
    clone.generation_kwargs = {"temperature": 0.6, "seed": 20260816}
    clone.call_tracker = 0
    clone.load_generation_state_dict(llm.generation_state_dict())
    clone.call_tracker += 1
    assert clone._generation_kwargs_for_call()["seed"] == 20260818


def test_evolutionary_checkpoint_includes_llm_call_schedule_state():
    node = make_node("a", 0.2)
    population = SolutionsDatabase(
        num_islands=1,
        max_size=10,
        lower_is_better=True,
        logger=NullLogger(),
    )
    population._islands[0].nodes = [node]
    population.global_min_fitness = 0.2
    population.global_max_fitness = 0.2

    llm = GenericLLM.__new__(GenericLLM)
    llm.generation_kwargs = {"seed": 123}
    llm.call_tracker = 4
    solver = Evolutionary.__new__(Evolutionary)
    solver.journal = type("JournalStub", (), {"nodes": [node]})()
    solver.state = EvolutionaryState()
    solver._operator_llms = {"draft": llm}

    payload = solver._population_checkpoint_payload(population)

    assert payload["schema_version"] == 2
    assert payload["llm_generation_state"] == {
        "draft": {"call_tracker": 4, "base_seed": 123}
    }


def make_population_checkpoint_restore_case():
    node = make_node("a", 0.2)
    population = SolutionsDatabase(
        num_islands=1,
        max_size=10,
        lower_is_better=True,
        logger=NullLogger(),
    )
    population._islands[0].nodes = [node]
    population.global_min_fitness = 0.2
    population.global_max_fitness = 0.2

    llm = GenericLLM.__new__(GenericLLM)
    llm.generation_kwargs = {"seed": 123}
    llm.call_tracker = 4

    source = Evolutionary.__new__(Evolutionary)
    source.journal = type("JournalStub", (), {"nodes": [node]})()
    source.state = EvolutionaryState()
    source.state.current_step = 5
    source.state.current_generation = 2
    source.state.running_time = 12.5
    source.state.num_starts = 0
    source._operator_llms = {"draft": llm}
    payload = source._population_checkpoint_payload(population)

    restored_population = SolutionsDatabase(
        num_islands=1,
        max_size=10,
        lower_is_better=True,
        logger=NullLogger(),
    )
    restored_llm = GenericLLM.__new__(GenericLLM)
    restored_llm.generation_kwargs = {"seed": 123}
    restored_llm.call_tracker = 0
    restored = Evolutionary.__new__(Evolutionary)
    restored.journal = type("JournalStub", (), {"nodes": [node]})()
    restored.state = EvolutionaryState()
    restored.state.load_state_dict(payload["solver_state"])
    # Solver.load_checkpoint deliberately records this new process start before
    # Evolutionary validates and restores the exact population checkpoint.
    restored.state.num_starts += 1
    restored._operator_llms = {"draft": restored_llm}
    return restored, restored_population, restored_llm, payload


def test_population_checkpoint_accepts_exact_state_after_one_restart():
    solver, population, llm, payload = make_population_checkpoint_restore_case()

    solver._load_population_checkpoint(population, payload)

    assert solver.state.num_starts == payload["solver_state"]["num_starts"] + 1
    assert [[node.id for node in island.nodes] for island in population._islands] == [["a"]]
    assert llm.call_tracker == 4


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("current_step", 6),
        ("current_generation", 3),
        ("running_time", 13.5),
        ("num_starts", 2),
    ],
)
def test_population_checkpoint_rejects_any_other_solver_state_mismatch(field, value):
    solver, population, _llm, payload = make_population_checkpoint_restore_case()
    setattr(solver.state, field, value)

    with pytest.raises(
        ValueError,
        match="Population checkpoint does not match the loaded solver state",
    ):
        solver._load_population_checkpoint(population, payload)


def test_population_checkpoint_rejects_missing_restart_provenance():
    solver, population, _llm, payload = make_population_checkpoint_restore_case()
    del payload["solver_state"]["num_starts"]

    with pytest.raises(
        ValueError,
        match="Population checkpoint is missing solver restart state",
    ):
        solver._load_population_checkpoint(population, payload)


def test_journal_round_trip_preserves_nested_metric_provenance_and_tie_order():
    journal = Journal()
    first = make_node("first", 0.2)
    first.metric.info = {
        "selection_score": 0.2,
        "selection_score_source": "sandbox_valid_score",
        "raw_scores": {"sandbox_valid_score": 0.2},
    }
    first.experience_parent_selection = {
        "selected_action_id": "improve:first",
        "forced_action": {"operator": "improve", "parent_node_ids": ["first"]},
    }
    second = make_node("second", 0.2)
    second.metric.info = {
        "selection_score": 0.2,
        "selection_score_source": "sandbox_valid_score",
    }
    journal.append(first)
    journal.append(second)

    restored = Journal.from_export_data({"nodes": journal.node_list()})

    assert [node.id for node in restored.nodes] == ["first", "second"]
    assert restored.nodes[0].metric.info == first.metric.info
    assert (
        restored.nodes[0].experience_parent_selection
        == first.experience_parent_selection
    )
    assert restored.nodes[1].metric.info == second.metric.info
    assert restored.get_best_node().id == "first"


def test_journal_round_trip_canonicalizes_children_and_preserves_worst_direction():
    journal = Journal()
    root = Node(
        code="",
        plan="",
        analysis="",
        metric=WorstMetricValue(
            maximize=False,
            info={"status": "failed", "failure_type": "engineering_probe"},
        ),
        is_buggy=True,
    )
    journal.append(root)
    first = make_node("first", 0.2)
    first.parents = [root]
    journal.append(first)
    second = make_node("second", 0.3)
    second.parents = [root]
    journal.append(second)

    exported = journal.node_list()
    restored = Journal.from_export_data({"nodes": exported})

    assert exported[0]["children"] == [1, 2]
    assert restored.node_list() == exported
    assert restored.nodes[0].metric.maximize is False
    assert restored.nodes[0].metric.info == root.metric.info
