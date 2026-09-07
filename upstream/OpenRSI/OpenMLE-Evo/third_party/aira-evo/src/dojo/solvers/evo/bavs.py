"""Budgeted Action-Value Selection for evolutionary code search.

The controller is intentionally lightweight: a regularized Bayesian linear model
predicts validity-adjusted improvement for complete Improve or ordered-Crossover
actions.  Program scores are never exposed to the generating model here.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy

from dojo.solvers.evo.experience import (
    _cosine_distance,
    _wl_features,
    detect_method_family,
    extract_imports,
)


POLICY_NAME = "bavs_v1"
FEATURE_NAMES = (
    "progress",
    "remaining",
    "log_support",
    "stagnation",
    "wl_spread",
    "is_crossover",
    "p1_score_component",
    "p1_score_gap",
    "p1_signed_delta",
    "p1_frontier_wl",
    "p1_family_rarity",
    "p1_age",
    "p1_selection_rate",
    "p1_mean_child_reward",
    "p1_child_uncertainty",
    "p2_score_component",
    "p2_score_gap",
    "p2_signed_delta",
    "p2_frontier_wl",
    "p2_family_rarity",
    "p2_age",
    "p2_selection_rate",
    "p2_mean_child_reward",
    "p2_child_uncertainty",
    "pair_score_gap",
    "pair_wl_distance",
    "same_family",
    "import_jaccard",
    "genealogically_related",
    "recent_operator_valid_rate",
    "recent_operator_gain_rate",
)


def _finite(value: Any) -> float | None:
    try:
        number = float(getattr(value, "value", value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _node_score(node: Any) -> float | None:
    return _finite(getattr(node, "metric", None))


def _is_valid(node: Any) -> bool:
    return not bool(getattr(node, "is_buggy", True)) and _node_score(node) is not None


def _operator(node: Any) -> str:
    used = list(getattr(node, "operators_used", None) or [])
    return str(used[0]).lower() if used else ""


def _action_nodes(nodes: Iterable[Any]) -> list[Any]:
    return [node for node in nodes if isinstance(getattr(node, "experience_parent_selection", None), dict)]


def _resolved_outcome(action_node: Any, nodes: Iterable[Any]) -> Any | None:
    if _is_valid(action_node):
        return action_node
    candidates = [
        node
        for node in nodes
        if _is_valid(node)
        and _operator(node) == "debug"
        and any(str(getattr(parent, "id", "")) == str(getattr(action_node, "id", "")) for parent in getattr(node, "parents", []) or [])
    ]
    return max(candidates, key=lambda node: int(getattr(node, "step", -1))) if candidates else None


def _score_scale(scores: list[float], incumbent: float) -> float:
    if len(scores) >= 2:
        q25, q75 = numpy.quantile(numpy.asarray(scores, dtype=float), [0.25, 0.75])
        iqr = float(q75 - q25)
    else:
        iqr = 0.0
    return max(iqr, 0.05 * max(abs(incumbent), 1e-6), 1e-6)


def _better(left: float, right: float, lower_is_better: bool) -> bool:
    return left < right if lower_is_better else left > right


def _improvement(before: float, after: float, scale: float, lower_is_better: bool) -> float:
    value = before - after if lower_is_better else after - before
    return float(numpy.clip(value / scale, -8.0, 8.0))


def _best(scores: list[float], lower_is_better: bool) -> float:
    return min(scores) if lower_is_better else max(scores)


def _normalize_scores(scores: list[float], lower_is_better: bool) -> list[float]:
    low, high = min(scores), max(scores)
    if low == high:
        return [0.5] * len(scores)
    if lower_is_better:
        return [(high - score) / (high - low) for score in scores]
    return [(score - low) / (high - low) for score in scores]


def _ancestors(node: Any) -> set[str]:
    found: set[str] = set()
    stack = list(getattr(node, "parents", None) or [])
    while stack:
        parent = stack.pop()
        parent_id = str(getattr(parent, "id", ""))
        if not parent_id or parent_id in found:
            continue
        found.add(parent_id)
        stack.extend(list(getattr(parent, "parents", None) or []))
    return found


def _related(left: Any, right: Any) -> bool:
    left_id, right_id = str(getattr(left, "id", "")), str(getattr(right, "id", ""))
    return left_id in _ancestors(right) or right_id in _ancestors(left)


def _imports(node: Any) -> set[str]:
    return set(extract_imports(str(getattr(node, "code", "") or "")))


def _history(nodes: list[Any], lower_is_better: bool) -> dict[str, Any]:
    valid_nodes = [node for node in nodes if _is_valid(node) and str(getattr(node, "code", "") or "").strip()]
    scores = [_node_score(node) for node in valid_nodes]
    finite_scores = [float(score) for score in scores if score is not None]
    incumbent = _best(finite_scores, lower_is_better) if finite_scores else 0.0
    scale = _score_scale(finite_scores, incumbent) if finite_scores else 1.0
    parent_rewards: dict[str, list[float]] = {}
    parent_selections: dict[str, int] = {}
    operator_rows: dict[str, list[tuple[bool, bool]]] = {"improve": [], "crossover": []}
    running_scores: list[float] = []
    last_gain_action = 0
    actions = _action_nodes(nodes)
    for action_index, action_node in enumerate(actions, start=1):
        event = dict(getattr(action_node, "experience_parent_selection", None) or {})
        operator = str(event.get("operator") or _operator(action_node)).lower()
        selected_ids = [str(value) for value in event.get("selected_node_ids") or []]
        prefix_scores = [
            float(score)
            for node in nodes
            if int(getattr(node, "step", -1)) < int(getattr(action_node, "step", -1))
            and (score := _node_score(node)) is not None
            and _is_valid(node)
        ]
        before = _best(prefix_scores, lower_is_better) if prefix_scores else incumbent
        outcome = _resolved_outcome(action_node, nodes)
        valid = outcome is not None
        gain = False
        reward = -1.0
        if valid:
            outcome_score = float(_node_score(outcome))
            action_scale = _score_scale(prefix_scores or finite_scores, before)
            reward = _improvement(before, outcome_score, action_scale, lower_is_better)
            gain = _better(outcome_score, before, lower_is_better)
            running_scores.append(outcome_score)
        if gain:
            last_gain_action = action_index
        if operator in operator_rows:
            operator_rows[operator].append((valid, gain))
        for parent_id in selected_ids:
            parent_selections[parent_id] = parent_selections.get(parent_id, 0) + 1
            parent_rewards.setdefault(parent_id, []).append(reward)
    return {
        "incumbent": incumbent,
        "scale": scale,
        "action_count": len(actions),
        "stagnation": max(0, len(actions) - last_gain_action),
        "parent_rewards": parent_rewards,
        "parent_selections": parent_selections,
        "operator_rows": operator_rows,
    }


def _support_features(support: list[Any], lower_is_better: bool, history: dict[str, Any]) -> dict[str, dict[str, float]]:
    scores = [float(_node_score(node)) for node in support]
    score_components = _normalize_scores(scores, lower_is_better)
    incumbent = _best(scores, lower_is_better)
    scale = _score_scale(scores, incumbent)
    vectors = _wl_features([str(getattr(node, "code", "") or "") for node in support])
    best_component = max(score_components)
    frontier = [vector for vector, component in zip(vectors, score_components) if best_component - component <= 1e-12]
    frontier_distances = [min(_cosine_distance(vector, leader) for leader in frontier) for vector in vectors]
    families = [detect_method_family(str(getattr(node, "code", "") or ""), extract_imports(str(getattr(node, "code", "") or ""))) for node in support]
    family_counts = {family: families.count(family) for family in set(families)}
    current_step = max([int(getattr(node, "step", 0)) for node in support] + [1])
    total_actions = max(1, int(history["action_count"]))
    result: dict[str, dict[str, float]] = {}
    for index, node in enumerate(support):
        node_id = str(getattr(node, "id", ""))
        parents = [parent for parent in getattr(node, "parents", []) or [] if _node_score(parent) is not None]
        signed_delta = 0.0
        if parents:
            parent_best = _best([float(_node_score(parent)) for parent in parents], lower_is_better)
            signed_delta = _improvement(parent_best, scores[index], scale, lower_is_better)
        rewards = list(history["parent_rewards"].get(node_id, []))
        result[node_id] = {
            "score_component": float(score_components[index]),
            "score_gap": float(_improvement(incumbent, scores[index], scale, not lower_is_better)),
            "signed_delta": signed_delta,
            "frontier_wl": float(frontier_distances[index]),
            "family_rarity": float(1.0 / math.sqrt(family_counts[families[index]])),
            "age": float(max(0, current_step - int(getattr(node, "step", 0))) / current_step),
            "selection_rate": float(history["parent_selections"].get(node_id, 0) / total_actions),
            "mean_child_reward": float(numpy.mean(rewards)) if rewards else 0.0,
            "child_uncertainty": float(1.0 / math.sqrt(1.0 + len(rewards))),
            "family": families[index],
            "vector_index": float(index),
        }
    result["__state__"] = {
        "wl_spread": float(max(frontier_distances) - min(frontier_distances)) if frontier_distances else 0.0,
        "incumbent": incumbent,
        "scale": scale,
        "vectors": vectors,
    }
    return result


def action_features(
    operator: str,
    parents: list[Any],
    support: list[Any],
    journal_prefix: list[Any],
    *,
    lower_is_better: bool,
    total_candidate_slots: int,
) -> list[float]:
    history = _history(journal_prefix, lower_is_better)
    support_info = _support_features(support, lower_is_better, history)
    state = support_info["__state__"]
    progress = min(1.0, history["action_count"] / max(1, total_candidate_slots))

    def parent_values(parent: Any | None) -> list[float]:
        if parent is None:
            return [0.0] * 9
        info = support_info[str(getattr(parent, "id", ""))]
        return [
            info["score_component"], info["score_gap"], info["signed_delta"],
            info["frontier_wl"], info["family_rarity"], info["age"],
            info["selection_rate"], info["mean_child_reward"], info["child_uncertainty"],
        ]

    first = parents[0]
    second = parents[1] if len(parents) > 1 else None
    op_rows = list(history["operator_rows"].get(operator, []))[-20:]
    recent_valid = sum(valid for valid, _ in op_rows) / len(op_rows) if op_rows else 0.5
    recent_gain = sum(gain for _, gain in op_rows) / len(op_rows) if op_rows else 0.1
    pair_values = [0.0] * 5
    if second is not None:
        first_info = support_info[str(getattr(first, "id", ""))]
        second_info = support_info[str(getattr(second, "id", ""))]
        vectors = state["vectors"]
        left_vector = vectors[int(first_info["vector_index"])]
        right_vector = vectors[int(second_info["vector_index"])]
        left_imports, right_imports = _imports(first), _imports(second)
        union = left_imports | right_imports
        pair_values = [
            abs(first_info["score_component"] - second_info["score_component"]),
            _cosine_distance(left_vector, right_vector),
            float(first_info["family"] == second_info["family"]),
            len(left_imports & right_imports) / len(union) if union else 1.0,
            float(_related(first, second)),
        ]
    values = [
        progress,
        1.0 - progress,
        math.log1p(len(support)),
        min(1.0, history["stagnation"] / 20.0),
        float(state["wl_spread"]),
        float(operator == "crossover"),
        *parent_values(first),
        *parent_values(second),
        *pair_values,
        float(recent_valid),
        float(recent_gain),
    ]
    if len(values) != len(FEATURE_NAMES):
        raise RuntimeError(f"BAVS feature length mismatch: {len(values)} != {len(FEATURE_NAMES)}")
    return [float(value) for value in values]


def _load_development_records(path_value: Any) -> tuple[list[dict[str, Any]], str | None]:
    if not path_value:
        return [], None
    path = Path(str(path_value))
    if not path.is_absolute():
        path = Path.cwd() / path
    raw = path.read_bytes()
    payload = json.loads(raw)
    if list(payload.get("feature_names") or []) != list(FEATURE_NAMES):
        raise ValueError("BAVS development feature schema mismatch")
    return list(payload.get("records") or []), hashlib.sha256(raw).hexdigest()


def extract_training_records(
    journal_nodes: list[Any],
    *,
    lower_is_better: bool,
    total_candidate_slots: int,
    source: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    all_nodes = list(journal_nodes)
    for action_node in _action_nodes(all_nodes):
        event = dict(getattr(action_node, "experience_parent_selection", None) or {})
        operator = str(event.get("operator") or _operator(action_node)).lower()
        if operator not in {"improve", "crossover"}:
            continue
        selected_ids = [str(value) for value in event.get("selected_node_ids") or []]
        expected = 2 if operator == "crossover" else 1
        if len(selected_ids) != expected:
            continue
        prefix = [node for node in all_nodes if int(getattr(node, "step", -1)) < int(getattr(action_node, "step", -1))]
        by_id = {str(getattr(node, "id", "")): node for node in prefix}
        candidate_ids = [str(item.get("node_id")) for item in event.get("candidates") or []]
        if not candidate_ids or any(node_id not in by_id for node_id in candidate_ids + selected_ids):
            continue
        support = [by_id[node_id] for node_id in candidate_ids]
        parents = [by_id[node_id] for node_id in selected_ids]
        features = action_features(
            operator, parents, support, prefix,
            lower_is_better=lower_is_better,
            total_candidate_slots=total_candidate_slots,
        )
        outcome = _resolved_outcome(action_node, all_nodes)
        valid = outcome is not None
        reward = None
        if valid:
            prefix_scores = [float(_node_score(node)) for node in prefix if _is_valid(node)]
            if not prefix_scores:
                continue
            incumbent = _best(prefix_scores, lower_is_better)
            scale = _score_scale(prefix_scores, incumbent)
            reward = _improvement(incumbent, float(_node_score(outcome)), scale, lower_is_better)
        records.append({
            "features": features,
            "valid": bool(valid),
            "reward": reward,
            "operator": operator,
            "source": source,
            "action_node_id": str(getattr(action_node, "id", "")),
        })
    return records


def _fit_ridge(records: list[dict[str, Any]], target: str, ridge: float) -> dict[str, Any]:
    selected = [row for row in records if row.get(target) is not None]
    x = numpy.asarray([row["features"] for row in selected], dtype=float)
    y = numpy.asarray([float(row[target]) for row in selected], dtype=float)
    if len(selected) < 2:
        raise ValueError(f"insufficient BAVS {target} records")
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-9] = 1.0
    z = (x - mean) / std
    design = numpy.column_stack([numpy.ones(len(z)), z])
    penalty = numpy.eye(design.shape[1]) * float(ridge)
    penalty[0, 0] = 0.0
    inverse = numpy.linalg.pinv(design.T @ design + penalty)
    weights = inverse @ design.T @ y
    residual = y - design @ weights
    variance = max(float(numpy.mean(residual**2)), 1e-4)
    return {"mean": mean, "std": std, "weights": weights, "inverse": inverse, "variance": variance, "count": len(selected)}


def _predict(model: dict[str, Any], features: list[list[float]]) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    x = numpy.asarray(features, dtype=float)
    z = (x - model["mean"]) / model["std"]
    design = numpy.column_stack([numpy.ones(len(z)), z])
    mean = design @ model["weights"]
    leverage = numpy.einsum("ij,jk,ik->i", design, model["inverse"], design)
    std = numpy.sqrt(model["variance"] * numpy.maximum(1.0 + leverage, 1e-9))
    return mean, std, design


def _expected_positive(mean: numpy.ndarray, std: numpy.ndarray) -> numpy.ndarray:
    std = numpy.maximum(std, 1e-9)
    z = mean / std
    phi = numpy.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    cdf = numpy.asarray([0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0))) for value in z])
    return std * phi + mean * cdf


def score_actions(
    actions: list[tuple[str, list[Any]]],
    support: list[Any],
    journal_nodes: list[Any],
    *,
    lower_is_better: bool,
    config: dict[str, Any],
    rng: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    total_slots = int(config.get("total_candidate_slots", 100))
    online = extract_training_records(
        journal_nodes,
        lower_is_better=lower_is_better,
        total_candidate_slots=total_slots,
        source="online",
    )
    development, development_hash = _load_development_records(config.get("development_data_path"))
    records = development + online
    features = [
        action_features(operator, parents, support, journal_nodes, lower_is_better=lower_is_better, total_candidate_slots=total_slots)
        for operator, parents in actions
    ]
    validity_ridge = float(config.get("validity_ridge", 4.0))
    reward_ridge = float(config.get("reward_ridge", 80.0))
    validity_records = [{**row, "valid_target": float(bool(row.get("valid")))} for row in records]
    validity_model = _fit_ridge(validity_records, "valid_target", validity_ridge)
    reward_model = _fit_ridge([row for row in records if row.get("valid")], "reward", reward_ridge)
    valid_mean, _, _ = _predict(validity_model, features)
    valid_probability = numpy.clip(valid_mean, 0.05, 0.98)
    reward_mean, reward_std, reward_design = _predict(reward_model, features)
    progress = _history(journal_nodes, lower_is_better)["action_count"] / max(1, total_slots)
    exploit = progress >= 1.0 - float(config.get("exploit_fraction", 0.2))
    sampled_mean = reward_mean
    if not exploit:
        covariance = reward_model["variance"] * reward_model["inverse"]
        sampled_weights = rng.multivariate_normal(reward_model["weights"], covariance)
        sampled_mean = reward_design @ sampled_weights
    acquisition = valid_probability * _expected_positive(sampled_mean, reward_std)
    rows = []
    for (operator, parents), p_valid, mean, std, sampled, value in zip(
        actions, valid_probability, reward_mean, reward_std, sampled_mean, acquisition
    ):
        parent_ids = [str(getattr(parent, "id", "")) for parent in parents]
        rows.append({
            "action_id": f"{operator}:" + ">".join(parent_ids),
            "operator": operator,
            "parent_node_ids": parent_ids,
            "predicted_valid_probability": float(p_valid),
            "predicted_reward_mean": float(mean),
            "predicted_reward_std": float(std),
            "sampled_reward_mean": float(sampled),
            "acquisition": float(value),
        })
    metadata = {
        "policy_name": POLICY_NAME,
        "feature_schema_sha256": hashlib.sha256("\n".join(FEATURE_NAMES).encode()).hexdigest(),
        "development_data_sha256": development_hash,
        "development_records": len(development),
        "online_records": len(online),
        "validity_training_records": validity_model["count"],
        "reward_training_records": reward_model["count"],
        "validity_ridge": validity_ridge,
        "reward_ridge": reward_ridge,
        "exploit": exploit,
    }
    return rows, metadata


def select_action(
    support: list[Any],
    journal_nodes: list[Any],
    *,
    lower_is_better: bool,
    allow_crossover: bool,
    config: dict[str, Any],
    rng: Any,
    decision_state_id: str | None = None,
) -> tuple[list[Any], str, dict[str, Any]]:
    actions: list[tuple[str, list[Any]]] = [("improve", [node]) for node in support]
    if allow_crossover and len(support) >= 2:
        actions.extend(
            ("crossover", [left, right])
            for left in support
            for right in support
            if str(getattr(left, "id", "")) != str(getattr(right, "id", ""))
        )
    scored, metadata = score_actions(
        actions, support, journal_nodes,
        lower_is_better=lower_is_better,
        config=config,
        rng=rng,
    )
    best_index = max(range(len(scored)), key=lambda index: (scored[index]["acquisition"], scored[index]["action_id"]))
    operator, parents = actions[best_index]
    selected = scored[best_index]
    candidates = []
    for node in support:
        candidates.append({
            "node_id": str(getattr(node, "id", "")),
            "fitness": float(_node_score(node)),
        })
    trace = {
        "enabled": True,
        "selection_policy": POLICY_NAME,
        "operator": operator,
        "num_in_context_samples": len(parents),
        "selected_node_ids": selected["parent_node_ids"],
        "selected_action_id": selected["action_id"],
        "selected_action_acquisition": selected["acquisition"],
        "candidates": candidates,
        "legal_actions": scored,
        "model": metadata,
        "active_pool_node_ids": [str(getattr(node, "id", "")) for node in support],
    }
    if decision_state_id is not None:
        trace["decision_state_id"] = str(decision_state_id)
    return parents, operator, trace


def select_active_pool(
    archive_nodes: list[Any],
    *,
    lower_is_better: bool,
    config: dict[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    candidates_by_id = {
        str(getattr(node, "id", "")): node
        for node in archive_nodes
        if _is_valid(node) and str(getattr(node, "code", "") or "").strip()
    }
    candidates = list(candidates_by_id.values())
    capacity = max(2, int(config.get("active_pool_size", 20)))
    elite_count = min(capacity, max(1, int(config.get("elite_count", 5))))
    potential_count = min(capacity - elite_count, max(0, int(config.get("potential_count", 10))))
    ordered = sorted(candidates, key=lambda node: float(_node_score(node)), reverse=not lower_is_better)
    if len(ordered) <= capacity:
        return ordered, {"policy_name": POLICY_NAME, "capacity": capacity, "selected_node_ids": [str(node.id) for node in ordered], "removed_node_ids": []}

    history = _history(archive_nodes, lower_is_better)
    selected = ordered[:elite_count]
    selected_ids = {str(node.id) for node in selected}
    potential_scores = {}
    for node in candidates:
        rewards = list(history["parent_rewards"].get(str(node.id), []))
        positive_mean = float(numpy.mean([max(0.0, reward) for reward in rewards])) if rewards else 0.0
        potential_scores[str(node.id)] = positive_mean + 1.0 / math.sqrt(1.0 + len(rewards))
    for node in sorted(candidates, key=lambda item: (potential_scores[str(item.id)], -int(getattr(item, "step", 0))), reverse=True):
        if len(selected) >= elite_count + potential_count:
            break
        if str(node.id) not in selected_ids:
            selected.append(node)
            selected_ids.add(str(node.id))

    quality_limit = max(capacity, int(math.ceil(0.75 * len(ordered))))
    diversity_candidates = [node for node in ordered[:quality_limit] if str(node.id) not in selected_ids]
    vectors = _wl_features([str(node.code or "") for node in candidates])
    vector_by_id = {str(node.id): vector for node, vector in zip(candidates, vectors)}
    while len(selected) < capacity and diversity_candidates:
        node = max(
            diversity_candidates,
            key=lambda item: (
                min(_cosine_distance(vector_by_id[str(item.id)], vector_by_id[str(chosen.id)]) for chosen in selected),
                -int(getattr(item, "step", 0)),
            ),
        )
        selected.append(node)
        selected_ids.add(str(node.id))
        diversity_candidates = [item for item in diversity_candidates if str(item.id) != str(node.id)]
    for node in ordered:
        if len(selected) >= capacity:
            break
        if str(node.id) not in selected_ids:
            selected.append(node)
            selected_ids.add(str(node.id))
    removed = [str(node.id) for node in candidates if str(node.id) not in selected_ids]
    return selected, {
        "policy_name": POLICY_NAME,
        "capacity": capacity,
        "elite_count": elite_count,
        "potential_count": potential_count,
        "selected_node_ids": [str(node.id) for node in selected],
        "removed_node_ids": removed,
        "potential_scores": potential_scores,
    }
