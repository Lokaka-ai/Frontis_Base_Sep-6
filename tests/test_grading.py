from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from frontis_local.sandbox.grading import EvaluationError, grade_submission


def test_spooky_grader_aligns_ids_without_exposing_answers(tmp_path: Path):
    answers = tmp_path / "answers.csv"
    submission = tmp_path / "submission.csv"
    pd.DataFrame(
        [
            {"id": "b", "EAP": 0, "HPL": 1, "MWS": 0},
            {"id": "a", "EAP": 1, "HPL": 0, "MWS": 0},
        ]
    ).to_csv(answers, index=False)
    pd.DataFrame(
        [
            {"id": "a", "EAP": 0.8, "HPL": 0.1, "MWS": 0.1},
            {"id": "b", "EAP": 0.1, "HPL": 0.8, "MWS": 0.1},
        ]
    ).to_csv(submission, index=False)
    score = grade_submission("spooky_log_loss", submission, answers)
    assert score == pytest.approx(-__import__("math").log(0.8))


def test_grader_rejects_identifier_mismatch(tmp_path: Path):
    answers = tmp_path / "answers.csv"
    submission = tmp_path / "submission.csv"
    pd.DataFrame([{"Id": 1, "PRICE": 100.0}]).to_csv(answers, index=False)
    pd.DataFrame([{"Id": 2, "PRICE": 100.0}]).to_csv(submission, index=False)
    with pytest.raises(EvaluationError, match="identifiers"):
        grade_submission("perth_rmsle", submission, answers)


def test_sms_grader_scores_aligned_binary_probabilities(tmp_path: Path):
    answers = tmp_path / "answers.csv"
    submission = tmp_path / "submission.csv"
    pd.DataFrame(
        [
            {"id": "b", "ham": 0, "spam": 1},
            {"id": "a", "ham": 1, "spam": 0},
        ]
    ).to_csv(answers, index=False)
    pd.DataFrame(
        [
            {"id": "a", "ham": 0.8, "spam": 0.2},
            {"id": "b", "ham": 0.1, "spam": 0.9},
        ]
    ).to_csv(submission, index=False)

    score = grade_submission("sms_binary_log_loss", submission, answers)

    assert score == pytest.approx(
        -(__import__("math").log(0.8) + __import__("math").log(0.9)) / 2
    )


def test_sms_grader_rejects_probability_rows_that_do_not_sum_to_one(tmp_path: Path):
    answers = tmp_path / "answers.csv"
    submission = tmp_path / "submission.csv"
    pd.DataFrame([{"id": "a", "ham": 1, "spam": 0}]).to_csv(answers, index=False)
    pd.DataFrame([{"id": "a", "ham": 0.8, "spam": 0.8}]).to_csv(submission, index=False)

    with pytest.raises(EvaluationError, match="probabilities"):
        grade_submission("sms_binary_log_loss", submission, answers)


def test_adult_grader_scores_probability_after_id_alignment(tmp_path: Path):
    answers = tmp_path / "answers.csv"
    submission = tmp_path / "submission.csv"
    pd.DataFrame(
        [
            {"id": "b", "probability_gt_50k": 1},
            {"id": "a", "probability_gt_50k": 0},
        ]
    ).to_csv(answers, index=False)
    pd.DataFrame(
        [
            {"id": "a", "probability_gt_50k": 0.2},
            {"id": "b", "probability_gt_50k": 0.9},
        ]
    ).to_csv(submission, index=False)

    score = grade_submission("adult_binary_log_loss", submission, answers)

    assert score == pytest.approx(
        -(__import__("math").log(0.8) + __import__("math").log(0.9)) / 2
    )


def test_adult_grader_rejects_out_of_range_probability(tmp_path: Path):
    answers = tmp_path / "answers.csv"
    submission = tmp_path / "submission.csv"
    pd.DataFrame([{"id": "a", "probability_gt_50k": 1}]).to_csv(answers, index=False)
    pd.DataFrame([{"id": "a", "probability_gt_50k": 1.1}]).to_csv(
        submission, index=False
    )

    with pytest.raises(EvaluationError, match="probabilities"):
        grade_submission("adult_binary_log_loss", submission, answers)


def test_bike_grader_scores_rmsle_after_id_alignment(tmp_path: Path):
    answers = tmp_path / "answers.csv"
    submission = tmp_path / "submission.csv"
    pd.DataFrame([{"instant": 2, "cnt": 9}, {"instant": 1, "cnt": 3}]).to_csv(
        answers, index=False
    )
    pd.DataFrame([{"instant": 1, "cnt": 3}, {"instant": 2, "cnt": 9}]).to_csv(
        submission, index=False
    )

    assert grade_submission("bike_rmsle", submission, answers) == pytest.approx(0.0)


def test_bike_grader_rejects_negative_predictions(tmp_path: Path):
    answers = tmp_path / "answers.csv"
    submission = tmp_path / "submission.csv"
    pd.DataFrame([{"instant": 1, "cnt": 3}]).to_csv(answers, index=False)
    pd.DataFrame([{"instant": 1, "cnt": -1}]).to_csv(submission, index=False)

    with pytest.raises(EvaluationError, match="predictions"):
        grade_submission("bike_rmsle", submission, answers)
