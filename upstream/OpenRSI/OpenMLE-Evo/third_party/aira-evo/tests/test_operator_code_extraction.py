import pytest

from dojo.core.solvers.operators.core import CodeExtractionError, execute_op_plan_code


PYTHON = "from pathlib import Path\nPath('submission.csv').write_text('id,y\\n1,0\\n')"


def test_execute_op_returns_extracted_python():
    calls = 0

    def operator():
        nonlocal calls
        calls += 1
        return f"```python\n{PYTHON}\n```", {"call": calls}

    plan, code, metrics = execute_op_plan_code(operator, max_operator_tries=1)

    assert plan == ""
    assert "submission.csv" in code
    assert metrics == {"call": 1}
    assert calls == 1


def test_execute_op_retries_then_accepts_valid_python():
    responses = iter(
        [
            '{"code": "print(1)"}{"code": "print(1)"}',
            f"```python\n{PYTHON}\n```",
        ]
    )

    plan, code, metrics = execute_op_plan_code(
        lambda: (next(responses), {}), max_operator_tries=2
    )

    assert plan == ""
    assert "submission.csv" in code
    assert metrics == {}


def test_execute_op_fails_closed_after_malformed_responses():
    malformed = '{"code": "print(1)"}{"code": "print(1)"}'

    with pytest.raises(CodeExtractionError, match="after 2 operator attempt") as raised:
        execute_op_plan_code(lambda: (malformed, {}), max_operator_tries=2)

    assert raised.value.attempt_metrics == [{}, {}]


def test_execute_op_rejects_zero_attempts():
    with pytest.raises(ValueError, match="at least 1"):
        execute_op_plan_code(lambda: (PYTHON, {}), max_operator_tries=0)
