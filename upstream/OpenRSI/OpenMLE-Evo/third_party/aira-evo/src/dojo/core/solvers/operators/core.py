# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

from dojo.core.solvers.utils.response import extract_code, extract_text_up_to_code, parse_thinking_tags


class CodeExtractionError(RuntimeError):
    """Raised when an operator never returns an extractable Python program."""

    def __init__(self, message: str, *, attempt_metrics: list[dict]):
        super().__init__(message)
        self.attempt_metrics = attempt_metrics


def execute_op_plan_code(
    operator_fn: Callable, *operator_args, max_operator_tries: int, requires_plan: bool = False
) -> tuple[str, str, str]:
    """Executes an operator function with the given arguments, attempts to extract the generated plan/code from the output
    and retries if the extraction fails."""
    if max_operator_tries < 1:
        raise ValueError("max_operator_tries must be at least 1")

    attempt_metrics = []
    for _ in range(max_operator_tries):
        completion_text, metrics = operator_fn(*operator_args)
        attempt_metrics.append(metrics)
        thinking_text, text_without_thinking = parse_thinking_tags(completion_text)
        code = extract_code(text_without_thinking)
        plan = extract_text_up_to_code(text_without_thinking)

        if code:
            if requires_plan and not plan:
                print("Plan extraction failed, retrying...")
                continue
            return plan, code, metrics

        print("Retrying Extraction...")

    raise CodeExtractionError(
        f"Failed to extract valid Python after {max_operator_tries} operator attempt(s)",
        attempt_metrics=attempt_metrics,
    )
