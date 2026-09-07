from pathlib import Path

import numpy

from dojo.core.solvers.utils.journal import Node
from dojo.core.solvers.utils.metric import MetricValue
from dojo.solvers.evo.bavs import select_action, select_active_pool
from dojo.solvers.evo.evo import SolutionsDatabase


class NullLogger:
    def info(self, *args, **kwargs):
        pass

    warning = error = debug = info


def make_node(index: int) -> Node:
    return Node(
        code=f"import numpy as np\ndef model_{index}(x):\n    return np.asarray(x) + {index}\n",
        id=f"n{index:02d}",
        step=index + 1,
        metric=MetricValue(float(index + 1), maximize=False),
        is_buggy=False,
        operators_used=["draft"],
    )


def config() -> dict:
    root = Path(__file__).resolve().parents[6]
    return {
        "development_data_path": str(root / "artifacts/analysis/bavs_v1_development_records.json"),
        "total_candidate_slots": 100,
        "active_pool_size": 20,
        "elite_count": 5,
        "potential_count": 10,
        "validity_ridge": 4.0,
        "reward_ridge": 80.0,
        "exploit_fraction": 0.2,
    }


def test_bavs_scores_every_complete_legal_action_and_selects_one():
    support = [make_node(index) for index in range(5)]
    parents, operator, trace = select_action(
        support,
        support,
        lower_is_better=True,
        allow_crossover=True,
        config=config(),
        rng=numpy.random.RandomState(9),
        decision_state_id="state-1",
    )

    assert len(trace["legal_actions"]) == 5 + 5 * 4
    assert operator in {"improve", "crossover"}
    assert len(parents) == (1 if operator == "improve" else 2)
    assert trace["selected_action_id"] in {row["action_id"] for row in trace["legal_actions"]}
    assert trace["decision_state_id"] == "state-1"


def test_bavs_active_pool_is_bounded_and_preserves_five_elites():
    archive = [make_node(index) for index in range(25)]
    selected, event = select_active_pool(archive, lower_is_better=True, config=config())

    assert len(selected) == 20
    assert len({node.id for node in selected}) == 20
    assert {f"n{index:02d}" for index in range(5)}.issubset({node.id for node in selected})
    assert len(event["removed_node_ids"]) == 5


def test_database_bavs_rebuilds_pool_from_archive():
    archive = [make_node(index) for index in range(25)]
    database = SolutionsDatabase(
        num_islands=1,
        max_size=500,
        lower_is_better=True,
        logger=NullLogger(),
        experience_config={
            "parent_selection": {"selection_policy": "bavs", "bavs": config()}
        },
    )
    database.journal_nodes = archive
    database._islands[0].nodes = archive[:5]
    database.add_nodes_to_islands(archive[5:], [0] * 20, migration_prob=0.0)

    assert len(database._islands[0].nodes) == 20
    assert database.bavs_population_events[-1]["capacity"] == 20
