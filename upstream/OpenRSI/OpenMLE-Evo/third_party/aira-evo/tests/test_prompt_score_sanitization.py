import sys
import types
from pathlib import Path

if "black" not in sys.modules:
    black_stub = types.ModuleType("black")
    black_stub.FileMode = object
    black_stub.format_str = lambda code, mode=None: code
    sys.modules["black"] = black_stub

from dojo.core.solvers.utils.response import prompt_score_sanitization_enabled
from dojo.core.solvers.utils.response import prompt_selection_metric_redaction_enabled
from dojo.core.solvers.utils.response import sanitize_execution_output_for_prompt
from dojo.core.solvers.operators.memory import create_memory_op, get_node_summary
from dojo.core.solvers.operators.rich_memory_summary import rich_memory_summary_op

OFFICIAL_SCORE_LOG = """Fold 1: 0.91
Final Validation Score: 0.9123

--- JigsawMetrics Submission Scorer ---
Reading prediction file...
Calculating score...
Final Score: 0.4567
prefix ##SCORE##0.4567 suffix

submission.csv Grader Feedback: ## Execution Result
**Status**: completed
**Score**: 0.4567
**Result**: success
"""


REPO_SRC = Path(__file__).resolve().parents[1] / "src"


def compact_source(text: str) -> str:
    return "".join(text.split())


def assert_official_score_removed(prompt_text: str):
    assert "Final Validation Score: 0.9123" in prompt_text
    assert "Final Score: 0.4567" not in prompt_text
    assert "##SCORE##0.4567" not in prompt_text
    assert "**Score**: 0.4567" not in prompt_text
    assert "Official sandbox score redacted" in prompt_text


def test_sanitizer_preserves_self_validation_and_removes_official_score():
    sanitized = sanitize_execution_output_for_prompt(OFFICIAL_SCORE_LOG)

    assert_official_score_removed(sanitized)


def test_sanitizer_can_be_disabled_for_inference_compatibility():
    assert sanitize_execution_output_for_prompt(OFFICIAL_SCORE_LOG, enabled=False) == OFFICIAL_SCORE_LOG


def test_prompt_score_sanitization_follows_experience_switch():
    assert prompt_score_sanitization_enabled({"experience": {"enabled": False}}) is False
    assert prompt_score_sanitization_enabled({}) is False
    assert prompt_score_sanitization_enabled({"experience": {"enabled": True}}) is True
    assert (
        prompt_score_sanitization_enabled(
            {
                "experience": {
                    "enabled": True,
                    "prompt_score_sanitization": {"enabled": False},
                }
            }
        )
        is False
    )


def test_selection_metric_redaction_is_separate_and_opt_in():
    assert prompt_selection_metric_redaction_enabled({}) is False
    assert (
        prompt_selection_metric_redaction_enabled(
            {
                "experience": {
                    "enabled": True,
                    "prompt_score_sanitization": {
                        "enabled": True,
                        "hide_selection_metric": True,
                    },
                }
            }
        )
        is True
    )


def test_base_memory_can_hide_node_metric_without_mutating_node():
    node = types.SimpleNamespace(
        plan="a plan",
        code="print('ok')",
        operators_used=["draft"],
        is_buggy=False,
        metric=types.SimpleNamespace(value=0.4567),
        analysis="completed",
    )
    hidden = get_node_summary(node, include_metric=False)
    visible = get_node_summary(node, include_metric=True)
    assert "0.4567" not in hidden
    assert "Validation Metric: 0.4567" in visible
    assert node.metric.value == 0.4567


def test_created_memory_operator_forwards_metric_visibility():
    node = types.SimpleNamespace(
        plan="a plan",
        code="print('ok')",
        operators_used=["draft"],
        is_buggy=False,
        metric=types.SimpleNamespace(value=0.4567),
        analysis="completed",
    )
    journal = types.SimpleNamespace(nodes=[node], good_nodes=[node])
    memory_cfg = types.SimpleNamespace(
        memory_processor="simple_memory",
        memory_op_kwargs={"only_plans": False, "include_buggy_nodes": False},
    )
    hidden_memory = create_memory_op(memory_cfg, include_metric=False)(journal)
    assert "0.4567" not in hidden_memory
    assert "a plan" in hidden_memory


def test_rich_memory_query_redacts_hidden_selection_metrics():
    captured = {}

    def fake_llm(**kwargs):
        captured.update(kwargs)
        return "{}"

    parent = types.SimpleNamespace(
        id="parent",
        metric=types.SimpleNamespace(value=0.8, info={"status": "success"}),
        experience_card={"fitness": 0.8, "status": "success"},
        operators_used=["draft"],
        plan="parent plan",
        code="print('parent')",
        term_out="",
        exec_time=1.0,
    )
    node = types.SimpleNamespace(
        id="node",
        metric=types.SimpleNamespace(value=0.4, info={"status": "success"}),
        experience_card={
            "fitness": 0.4,
            "delta_vs_parent": 0.4,
            "status": "success",
        },
        operators_used=["improve"],
        plan="node plan",
        code="print('node')",
        term_out="",
        exec_time=1.0,
    )
    cfg = {
        "experience": {
            "enabled": True,
            "prompt_score_sanitization": {
                "enabled": True,
                "hide_selection_metric": True,
            },
        }
    }
    rich_memory_summary_op(fake_llm, cfg, "task", node, parent)
    query = captured["query_data"]
    assert query["score"] == "redacted_hidden_selection_metric"
    assert query["delta_vs_parent"] == "redacted_hidden_selection_metric"
    assert query["parent_score"] == "redacted_hidden_selection_metric"
    assert node.metric.value == 0.4
    assert parent.metric.value == 0.8


def test_operator_prompt_sources_sanitize_term_out_before_prompting():
    source_expectations = {
        "dojo/core/solvers/operators/analyze.py": [
            "execution_output=sanitize_execution_output_for_prompt(input_node.term_out,enabled=prompt_score_sanitization_enabled(cfg),)",
        ],
        "dojo/core/solvers/operators/improve.py": [
            'sanitize_execution_output_for_prompt(input_node.term_out,enabled=sanitize_prompt_scores,)',
        ],
        "dojo/core/solvers/operators/crossover.py": [
            'sanitize_execution_output_for_prompt(input_node1.term_out,enabled=sanitize_prompt_scores,)',
            'sanitize_execution_output_for_prompt(input_node2.term_out,enabled=sanitize_prompt_scores,)',
        ],
        "dojo/core/solvers/operators/debug.py": [
            "execution_output=sanitize_execution_output_for_prompt(input_node.term_out,enabled=sanitize_prompt_scores,)",
        ],
        "dojo/core/solvers/operators/rich_memory_summary.py": [
            'sanitize_execution_output_for_prompt(getattr(input_node,"term_out","")or"",enabled=sanitize_prompt_scores,)',
            "sanitize_execution_output_for_prompt(",
        ],
    }

    for rel_path, snippets in source_expectations.items():
        source = compact_source((REPO_SRC / rel_path).read_text())
        assert "sanitize_execution_output_for_prompt" in source
        for snippet in snippets:
            assert compact_source(snippet) in source


def test_rich_memory_prompt_source_uses_selection_metric_only_when_not_redacted():
    source = compact_source(
        (REPO_SRC / "dojo/core/solvers/operators/rich_memory_summary.py").read_text()
    )

    assert compact_source('"score": (') in source
    assert compact_source('else _value(current_card.get("fitness"),') in source
    assert compact_source('"parent_score": (') in source
    assert compact_source('else _value(parent_card.get("fitness"),') in source
    assert compact_source('current_card.get("score")') not in source
    assert compact_source('parent_card.get("score")') not in source
    assert compact_source('current_info.get("score")') not in source
    assert compact_source('parent_info.get("score")') not in source
