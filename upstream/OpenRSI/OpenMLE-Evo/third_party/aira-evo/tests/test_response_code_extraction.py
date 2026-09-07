from dojo.core.solvers.utils.response import extract_code


PYTHON = "from pathlib import Path\nPath('submission.csv').write_text('id,y\\n1,0\\n')"


def test_extract_code_preserves_plain_python():
    assert "submission.csv" in extract_code(PYTHON)


def test_extract_code_preserves_python_fence():
    assert "submission.csv" in extract_code(f"```python\n{PYTHON}\n```")


def test_extract_code_unwraps_json_code_field():
    import json

    assert "submission.csv" in extract_code(json.dumps({"code": PYTHON}))


def test_extract_code_unwraps_json_solution_field():
    import json

    assert "submission.csv" in extract_code(json.dumps({"solution": PYTHON}))


def test_extract_code_unwraps_json_fence():
    import json

    response = f"```json\n{json.dumps({'code': PYTHON})}\n```"
    assert "submission.csv" in extract_code(response)


def test_extract_code_rejects_malformed_json_envelope():
    assert extract_code('{"code": "print(1)",}') == ""


def test_extract_code_rejects_non_code_json_envelope():
    assert extract_code('{"explanation": "no candidate"}') == ""


def test_extract_code_rejects_conflicting_code_fields():
    assert extract_code('{"code": "print(1)", "solution": "print(2)"}') == ""
