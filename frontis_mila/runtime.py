"""Observational hooks and explicit attempt-boundary budget for this protocol."""

from __future__ import annotations
import functools
import json
import os
from pathlib import Path
from frontis_mila.recording import event


class BudgetComplete(RuntimeError):
    """Clean stop at a committed slot boundary, also used for requested pause."""


def install():
    from dojo.solvers.evo.evo import (
        Evolutionary,
        SolutionsDatabase,
        Island,
        capture_rng_state,
    )

    if getattr(Evolutionary, "_mila_hooks_installed", False):
        return
    Evolutionary._mila_hooks_installed = True
    begin = Evolutionary._begin_candidate_slot

    @functools.wraps(begin)
    def before_slot(self, generation, individual, task=None):
        run = Path(os.environ["FRONTIS_RUN_DIR"])
        reason = "pause_requested" if (run / "PAUSE").exists() else None
        if self.state.running_time >= self.cfg.time_limit_secs:
            reason = "budget_exhausted"
        if reason:
            event("boundary_stop", reason=reason, used_seconds=self.state.running_time)
            raise BudgetComplete(reason)
        result = begin(self, generation, individual, task)
        event(
            "slot_begin",
            generation=generation,
            individual=individual,
            journal_step=self.state.current_step,
            rng_state=capture_rng_state(),
        )
        if generation == 0:
            self._save_parent_decision_state(
                self.solution_database,
                generation_id=generation,
                individual_id=individual,
            )
            event(
                "selection",
                trace={
                    "operator": "draft",
                    "reason": "initialization",
                    "operator_probabilities": {"draft": 1.0},
                    "selected_node_ids": [],
                },
            )
        return result

    Evolutionary._begin_candidate_slot = before_slot

    sample = SolutionsDatabase.sample_in_context

    @functools.wraps(sample)
    def selected(
        self,
        num_samples,
        temperature,
        crossover_prob,
        fresh_draft_prob=0.0,
        decision_state_id=None,
    ):
        sizes = [i.size for i in self._islands]
        forced = getattr(self, "_forced_fresh_draft_reason", None)
        result = sample(
            self,
            num_samples,
            temperature,
            crossover_prob,
            fresh_draft_prob,
            decision_state_id,
        )
        # This protocol is one-island, zero spontaneous draft. The fallback is explicit.
        if forced or not any(sizes):
            op_p = {"draft": 1.0, "improve": 0.0, "crossover": 0.0}
        elif max(sizes) < num_samples.get("crossover", 2):
            op_p = {"draft": 0.0, "improve": 1.0, "crossover": 0.0}
        else:
            op_p = {
                "draft": 0.0,
                "improve": 1.0 - crossover_prob,
                "crossover": crossover_prob,
            }
        trace = dict(self.last_parent_selection or {})
        trace.update(
            operator_probabilities=op_p,
            requested_crossover_probability=crossover_prob,
            island_sizes=sizes,
            island_probabilities=[1.0],
            decision_state_id=decision_state_id,
            selected_node_ids=[str(n.id) for n in result[0]],
            operator=result[2],
        )
        self.last_parent_selection = trace
        event("selection", trace=trace)
        return result

    SolutionsDatabase.sample_in_context = selected

    for method in ("seed_islands_with_nodes", "add_nodes_to_islands"):
        original = getattr(SolutionsDatabase, method)

        def wrap(fn, name):
            @functools.wraps(fn)
            def call(self, nodes, islands, *args, **kwargs):
                before = self.state_dict()
                offered = [
                    {"node_id": str(n.id), "island_id": i, "fitness": n.metric.value}
                    for n, i in zip(nodes, islands)
                ]
                result = fn(self, nodes, islands, *args, **kwargs)
                event(
                    "population_update",
                    operation=name,
                    offered=offered,
                    before=before,
                    after=self.state_dict(),
                )
                return result

            return call

        setattr(SolutionsDatabase, method, wrap(original, method))

    for method in (
        "register_node_in_island",
        "remove_lowest",
        "remove_node",
        "only_keep_best",
        "migrate_node",
    ):
        original = getattr(Island, method)

        def wrap_island(fn, name):
            @functools.wraps(fn)
            def call(self, *args, **kwargs):
                before = [str(n.id) for n in self.nodes]
                result = fn(self, *args, **kwargs)
                event(
                    "island_mutation",
                    operation=name,
                    island_id=self.island_id,
                    before=before,
                    after=[str(n.id) for n in self.nodes],
                )
                return result

            return call

        setattr(Island, method, wrap_island(original, method))
