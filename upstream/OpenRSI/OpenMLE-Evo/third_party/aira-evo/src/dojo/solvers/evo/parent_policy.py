"""Explicit legal-action distributions for evolutionary parent selection.

The Frontis controller assigns a probability to each node and samples parents
without replacement.  This module makes the induced action distribution
auditable and keeps sampling and probability reporting on the same code path.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy


POLICY_SCHEMA_VERSION = 1


def normalize_probabilities(weights: Sequence[float]) -> list[float]:
    """Return a finite probability vector, using uniform mass if necessary."""
    values = numpy.asarray(list(weights), dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("parent weights must be a non-empty one-dimensional sequence")
    if not numpy.isfinite(values).all() or (values < 0).any():
        raise ValueError("parent weights must be finite and non-negative")
    total = float(values.sum())
    if total <= 0:
        values = numpy.full(values.shape, 1.0 / values.size, dtype=float)
    else:
        values = values / total
    return [float(value) for value in values]


def sample_without_replacement(
    probabilities: Sequence[float],
    sample_size: int,
    *,
    rng: Any = numpy.random,
) -> list[int]:
    """Sequentially sample indices without replacement from explicit weights.

    The sequential implementation is intentional: it defines the exact ordered
    action probability reported by :func:`enumerate_legal_actions`.
    """
    remaining_indices = list(range(len(probabilities)))
    remaining_weights = normalize_probabilities(probabilities)
    if sample_size < 0 or sample_size > len(remaining_indices):
        raise ValueError("sample_size is outside the legal parent support")
    selected: list[int] = []
    for _ in range(sample_size):
        local_probabilities = normalize_probabilities(remaining_weights)
        local_index = int(rng.choice(len(remaining_indices), p=local_probabilities))
        selected.append(remaining_indices.pop(local_index))
        remaining_weights.pop(local_index)
    return selected


def _action_id(operator: str, parent_ids: Sequence[str]) -> str:
    return f"{operator}:" + "->".join(parent_ids)


def enumerate_legal_actions(
    operator: str,
    node_ids: Sequence[str],
    probabilities: Sequence[float],
) -> list[dict[str, Any]]:
    """Enumerate the exact legal Improve or ordered-Crossover actions."""
    ids = [str(node_id) for node_id in node_ids]
    probs = normalize_probabilities(probabilities)
    if len(ids) != len(probs) or len(set(ids)) != len(ids):
        raise ValueError("node IDs and probabilities must form a unique aligned support")
    if operator == "improve":
        actions = [
            {
                "action_id": _action_id(operator, [node_id]),
                "parent_node_ids": [node_id],
                "probability": probability,
            }
            for node_id, probability in zip(ids, probs)
        ]
    elif operator == "crossover":
        if len(ids) < 2:
            raise ValueError("crossover requires at least two legal parents")
        actions = []
        for first_index, first_id in enumerate(ids):
            remaining_mass = 1.0 - probs[first_index]
            if remaining_mass <= 0:
                continue
            for second_index, second_id in enumerate(ids):
                if first_index == second_index:
                    continue
                probability = probs[first_index] * probs[second_index] / remaining_mass
                actions.append(
                    {
                        "action_id": _action_id(operator, [first_id, second_id]),
                        "parent_node_ids": [first_id, second_id],
                        "probability": float(probability),
                    }
                )
    else:
        raise ValueError(f"unsupported parent-selection operator: {operator}")
    total = sum(float(action["probability"]) for action in actions)
    if actions and not numpy.isclose(total, 1.0, atol=1e-12):
        raise AssertionError(f"legal-action probabilities sum to {total}, not one")
    return actions


def build_action_audit(
    *,
    operator: str,
    node_ids: Sequence[str],
    probabilities: Sequence[float],
    selected_node_ids: Sequence[str],
    policy_name: str = "frontis_fixed_utility",
) -> dict[str, Any]:
    """Build the auditable legal support and identify the sampled action."""
    if os.environ.get("FRONTIS_RUN_DIR") and policy_name == "frontis_fixed_utility":
        # Lossless O(n) representation of the full O(n^2) pair distribution.
        ids = [str(i) for i in node_ids]
        p = normalize_probabilities(probabilities)
        chosen = [str(i) for i in selected_node_ids]
        size = 1 if operator == "improve" else 2
        if operator not in {"improve", "crossover"} or len(chosen) != size or len(set(chosen)) != size or not set(chosen) <= set(ids):
            raise ValueError("Invalid selected action")
        probability = p[ids.index(chosen[0])]
        if size == 2:
            probability *= p[ids.index(chosen[1])] / (1.0 - probability)
        return {
            "policy_schema_version": 2, "policy_name": policy_name,
            "action_semantics": "single_parent" if size == 1 else "ordered_parent_pair",
            "legal_action_distribution": {
                "encoding": "sequential_weighted_without_replacement_v1",
                "node_ids": ids, "probabilities": p, "sample_size": size,
                "legal_action_count": len(ids) if size == 1 else len(ids)*(len(ids)-1),
            },
            "selected_action_id": _action_id(operator, chosen),
            "selected_action_probability": probability,
        }
    actions = enumerate_legal_actions(operator, node_ids, probabilities)
    selected_ids = [str(node_id) for node_id in selected_node_ids]
    selected_action_id = _action_id(operator, selected_ids)
    if selected_action_id not in {action["action_id"] for action in actions}:
        raise ValueError("sampled parents are outside the enumerated legal action support")
    return {
        "policy_schema_version": POLICY_SCHEMA_VERSION,
        "policy_name": str(policy_name),
        "action_semantics": (
            "single_parent" if operator == "improve" else "ordered_parent_pair"
        ),
        "legal_actions": actions,
        "selected_action_id": selected_action_id,
    }


def sample_protected_elite(
    *, operator: str, node_ids: Sequence[str], probabilities: Sequence[float],
    fitness: Sequence[float], lower_is_better: bool, elite_count: int = 5,
    protected_mass: float = 0.6, rng: Any = numpy.random,
) -> tuple[list[int], dict[str, Any]]:
    """KL-minimal protection of elite-containing actions, conditional on operator.

    Keep the baseline sampler (including RNG consumption) when the constraint is
    inactive. Active pair sampling uses the exact joint's first marginal and
    conditional second distribution, retaining two draws and ordered semantics.
    """
    if elite_count < 1 or not 0 <= protected_mass < 1:
        raise ValueError('invalid protected-elite parameters')
    ids = list(node_ids)
    scores = numpy.asarray(fitness, dtype=float)
    if len(scores) != len(ids) or not numpy.isfinite(scores).all():
        raise ValueError('protected elites require finite aligned common fitness')
    p = normalize_probabilities(probabilities)
    if min(p) <= 0:
        raise ValueError('protected elites require positive baseline support')
    actions = enumerate_legal_actions(operator, ids, p)
    ordered = sorted(scores, reverse=not lower_is_better)
    cutoff = ordered[min(elite_count, len(scores))-1]
    elite_ids = {node_id for node_id, value in zip(ids, scores)
                 if (value <= cutoff if lower_is_better else value >= cutoff)}
    flags = [bool(elite_ids.intersection(a['parent_node_ids'])) for a in actions]
    mass = sum(a['probability'] for a, flag in zip(actions, flags) if flag)
    active = mass < protected_mass
    for a, flag in zip(actions, flags):
        a['baseline_probability'] = a['probability']
        a['protected'] = flag
        if active:
            a['probability'] *= protected_mass/mass if flag else (1-protected_mass)/(1-mass)
    if not active:
        indices = sample_without_replacement(p, 1 if operator == 'improve' else 2, rng=rng)
    elif operator == 'improve':
        indices = [int(rng.choice(len(ids), p=normalize_probabilities([a['probability'] for a in actions])))]
    else:
        index_by_id = {node_id:i for i,node_id in enumerate(ids)}
        first_mass = [0.0]*len(ids)
        for a in actions:
            first_mass[index_by_id[a['parent_node_ids'][0]]] += a['probability']
        first = int(rng.choice(len(ids), p=normalize_probabilities(first_mass)))
        second_indices = [i for i in range(len(ids)) if i != first]
        pairs = {tuple(a['parent_node_ids']):a['probability'] for a in actions}
        conditional = normalize_probabilities([pairs[(ids[first],ids[j])] for j in second_indices])
        second = second_indices[int(rng.choice(len(second_indices),p=conditional))]
        indices = [first,second]
    return indices, {
        'policy_schema_version': POLICY_SCHEMA_VERSION,
        'policy_name': 'protected_elite_v1', 'selection_policy': 'protected_elite',
        'action_semantics': 'single_parent' if operator == 'improve' else 'ordered_parent_pair',
        'elite_node_ids': [node_id for node_id in ids if node_id in elite_ids],
        'elite_count': elite_count, 'protected_mass_target': protected_mass,
        'baseline_protected_mass': mass, 'constraint_active': active,
        'protected_mass': max(mass,protected_mass),
        'legal_actions': actions,
        'selected_action_id': _action_id(operator,[ids[i] for i in indices]),
    }
