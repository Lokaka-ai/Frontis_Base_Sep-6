"""Host-only graders. This module is never mounted into candidate containers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


class EvaluationError(RuntimeError):
    pass


class EvaluationInfrastructureError(EvaluationError):
    pass


def _spooky_log_loss(submission_path: Path, answers_path: Path) -> float:
    expected = ["id", "EAP", "HPL", "MWS"]
    submission = pd.read_csv(submission_path)
    answers = pd.read_csv(answers_path)
    if list(submission.columns) != expected or list(answers.columns) != expected:
        raise EvaluationError("submission or answer schema is invalid")
    if submission.shape != answers.shape or submission["id"].duplicated().any():
        raise EvaluationError("submission dimensions or identifiers are invalid")
    if set(submission["id"]) != set(answers["id"]):
        raise EvaluationError("submission identifiers do not match")
    aligned = submission.set_index("id").loc[answers["id"]]
    probabilities = aligned[["EAP", "HPL", "MWS"]].to_numpy(dtype=float)
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise EvaluationError("submission probabilities are invalid")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise EvaluationError("submission probability rows do not sum to one")
    labels = answers[["EAP", "HPL", "MWS"]].to_numpy().argmax(axis=1)
    score = float(log_loss(labels, probabilities, labels=[0, 1, 2]))
    if not math.isfinite(score):
        raise EvaluationError("grader returned a non-finite score")
    return score


def _perth_rmsle(submission_path: Path, answers_path: Path) -> float:
    expected = ["Id", "PRICE"]
    submission = pd.read_csv(submission_path)
    answers = pd.read_csv(answers_path)
    if list(submission.columns) != expected or list(answers.columns) != expected:
        raise EvaluationError("submission or answer schema is invalid")
    if len(submission) != len(answers) or submission["Id"].duplicated().any():
        raise EvaluationError("submission dimensions or identifiers are invalid")
    if set(submission["Id"]) != set(answers["Id"]):
        raise EvaluationError("submission identifiers do not match")
    aligned = submission.set_index("Id").loc[answers["Id"]]
    predictions = pd.to_numeric(aligned["PRICE"], errors="coerce").to_numpy(dtype=float)
    targets = pd.to_numeric(answers["PRICE"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(predictions).all() or (predictions <= 0).any():
        raise EvaluationError("PRICE predictions are invalid")
    score = float(np.sqrt(np.mean((np.log1p(targets) - np.log1p(predictions)) ** 2)))
    if not math.isfinite(score):
        raise EvaluationError("grader returned a non-finite score")
    return score


def _binary_probability_log_loss(
    submission_path: Path,
    answers_path: Path,
    *,
    id_column: str,
    probability_column: str,
) -> float:
    expected = [id_column, probability_column]
    submission = pd.read_csv(submission_path)
    answers = pd.read_csv(answers_path)
    if list(submission.columns) != expected or list(answers.columns) != expected:
        raise EvaluationError("submission or answer schema is invalid")
    if len(submission) != len(answers) or submission[id_column].duplicated().any():
        raise EvaluationError("submission dimensions or identifiers are invalid")
    if set(submission[id_column]) != set(answers[id_column]):
        raise EvaluationError("submission identifiers do not match")
    aligned = submission.set_index(id_column).loc[answers[id_column]]
    probabilities = pd.to_numeric(
        aligned[probability_column], errors="coerce"
    ).to_numpy(dtype=float)
    targets = pd.to_numeric(answers[probability_column], errors="raise").to_numpy(
        dtype=int
    )
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not set(np.unique(targets)).issubset({0, 1})
    ):
        raise EvaluationError("binary probabilities or targets are invalid")
    score = float(log_loss(targets, probabilities, labels=[0, 1]))
    if not math.isfinite(score):
        raise EvaluationError("grader returned a non-finite score")
    return score


def _sms_binary_log_loss(submission_path: Path, answers_path: Path) -> float:
    expected = ["id", "ham", "spam"]
    submission = pd.read_csv(submission_path)
    answers = pd.read_csv(answers_path)
    if list(submission.columns) != expected or list(answers.columns) != expected:
        raise EvaluationError("submission or answer schema is invalid")
    if len(submission) != len(answers) or submission["id"].duplicated().any():
        raise EvaluationError("submission dimensions or identifiers are invalid")
    if set(submission["id"]) != set(answers["id"]):
        raise EvaluationError("submission identifiers do not match")
    aligned = submission.set_index("id").loc[answers["id"]]
    probabilities = (
        aligned[["ham", "spam"]]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise EvaluationError("submission probabilities are invalid")
    targets = answers[["ham", "spam"]].to_numpy(dtype=int).argmax(axis=1)
    score = float(log_loss(targets, probabilities, labels=[0, 1]))
    if not math.isfinite(score):
        raise EvaluationError("grader returned a non-finite score")
    return score


def _bike_rmsle(submission_path: Path, answers_path: Path) -> float:
    expected = ["instant", "cnt"]
    submission = pd.read_csv(submission_path)
    answers = pd.read_csv(answers_path)
    if list(submission.columns) != expected or list(answers.columns) != expected:
        raise EvaluationError("submission or answer schema is invalid")
    if len(submission) != len(answers) or submission["instant"].duplicated().any():
        raise EvaluationError("submission dimensions or identifiers are invalid")
    if set(submission["instant"]) != set(answers["instant"]):
        raise EvaluationError("submission identifiers do not match")
    aligned = submission.set_index("instant").loc[answers["instant"]]
    predictions = pd.to_numeric(aligned["cnt"], errors="coerce").to_numpy(dtype=float)
    targets = pd.to_numeric(answers["cnt"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(predictions).all() or (predictions < 0).any():
        raise EvaluationError("cnt predictions are invalid")
    score = float(np.sqrt(np.mean((np.log1p(targets) - np.log1p(predictions)) ** 2)))
    if not math.isfinite(score):
        raise EvaluationError("grader returned a non-finite score")
    return score


GRADERS: dict[str, Callable[[Path, Path], float]] = {
    "spooky_log_loss": _spooky_log_loss,
    "perth_rmsle": _perth_rmsle,
    "sms_binary_log_loss": _sms_binary_log_loss,
    "adult_binary_log_loss": lambda submission, answers: _binary_probability_log_loss(
        submission,
        answers,
        id_column="id",
        probability_column="probability_gt_50k",
    ),
    "bike_rmsle": _bike_rmsle,
}


def grade_submission(grader: str, submission_path: Path, answers_path: Path) -> float:
    try:
        implementation = GRADERS[grader]
    except KeyError as exc:
        raise EvaluationInfrastructureError(f"unknown grader: {grader}") from exc
    if not answers_path.is_file():
        raise EvaluationInfrastructureError("hidden answer file is unavailable")
    return implementation(submission_path, answers_path)
