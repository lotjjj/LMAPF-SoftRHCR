from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np


def agent_index(agent_id: str) -> int:
    return int(str(agent_id).split("_")[-1])


def agent_name_from_index(agent_idx: int) -> str:
    return f"agv_{int(agent_idx)}"


def extract_planner_paths(info: Optional[Dict[str, Any]], agent_id: str) -> Mapping[str, Any]:
    if not isinstance(info, dict):
        return {}
    item = info.get(agent_id)
    if not isinstance(item, Mapping):
        return {}
    planner_paths = item.get("planner_paths")
    if not isinstance(planner_paths, Mapping):
        return {}
    return planner_paths


def extract_shared_planner_paths(info: Optional[Dict[str, Any]]) -> Mapping[str, Any]:
    if not isinstance(info, dict):
        return {}
    for item in info.values():
        if not isinstance(item, Mapping):
            continue
        planner_paths = item.get("planner_paths")
        if isinstance(planner_paths, Mapping):
            return planner_paths
    return {}


def extract_planner_meta(info: Optional[Dict[str, Any]], agent_id: str) -> Dict[str, Any]:
    if not isinstance(info, dict):
        return {}
    item = info.get(agent_id)
    if not isinstance(item, Mapping):
        return {}
    planner_meta = item.get("planner_meta")
    if not isinstance(planner_meta, Mapping):
        return {}
    return dict(planner_meta)


def current_pos_from_obs(agent_obs: Mapping[str, Any], map_width: int, map_height: int) -> tuple[int, int]:
    ss = agent_obs.get("self_states")
    if not isinstance(ss, Mapping):
        return 0, 0
    pos = np.asarray(ss.get("position", np.zeros((2,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    if pos.size < 2:
        return 0, 0
    x = int(round(float(pos[0]) * float(max(0, map_width - 1)))) if map_width > 1 else 0
    y = int(round(float(pos[1]) * float(max(0, map_height - 1)))) if map_height > 1 else 0
    return x, y


def path_array_from_entry(
    entry: Optional[Mapping[str, Any]],
    fallback_pos: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    fallback = (0, 0) if fallback_pos is None else fallback_pos
    if not isinstance(entry, Mapping):
        return np.asarray([[float(fallback[0]), float(fallback[1])]], dtype=np.float32)
    path = np.asarray(entry.get("path_abs", []), dtype=np.float32)
    if path.ndim != 2 or path.shape[1] != 2 or path.shape[0] <= 0:
        return np.asarray([[float(fallback[0]), float(fallback[1])]], dtype=np.float32)
    return path.astype(np.float32, copy=False)


def hold_last_path(path_abs: np.ndarray, k_horizon: int) -> np.ndarray:
    """Pad a finite planner path by repeating its terminal position up to horizon K."""
    if not isinstance(path_abs, np.ndarray) or path_abs.ndim != 2 or path_abs.shape[1] != 2 or path_abs.shape[0] <= 0:
        return np.zeros((1, 2), dtype=np.float32)
    seq = path_abs.astype(np.float32, copy=False)
    target_len = max(1, int(k_horizon) + 1)
    if int(seq.shape[0]) >= target_len:
        return seq
    tail = np.repeat(seq[-1:, :], target_len - int(seq.shape[0]), axis=0)
    return np.concatenate([seq, tail], axis=0)


def has_planner_path(entry: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(entry, Mapping):
        return False
    if "has_path" in entry:
        return bool(entry.get("has_path", False))
    path = np.asarray(entry.get("path_abs", []), dtype=np.float32)
    return bool(path.ndim == 2 and path.shape[0] >= 2 and path.shape[1] == 2)


def build_path_info(
    path_abs: np.ndarray,
    current_pos: tuple[int, int],
    k_horizon: int,
    map_width: int,
    map_height: int,
    has_path: bool,
) -> np.ndarray:
    out = np.zeros((int(k_horizon) * 2 + 1,), dtype=np.float32)
    out[0] = 1.0 if bool(has_path) else 0.0
    if not bool(has_path):
        return out
    seq = hold_last_path(path_abs, int(k_horizon))
    if seq.ndim != 2 or seq.shape[0] <= 1:
        return out
    max_dx = float(max(1, map_width - 1))
    max_dy = float(max(1, map_height - 1))
    steps = min(int(k_horizon), int(seq.shape[0]) - 1)
    for i in range(steps):
        step_pos = seq[i + 1]
        out[i * 2 + 1] = float(step_pos[0] - float(current_pos[0])) / max_dx if map_width > 1 else 0.0
        out[i * 2 + 2] = float(step_pos[1] - float(current_pos[1])) / max_dy if map_height > 1 else 0.0
    return out


def round_pair(pair: np.ndarray) -> tuple[int, int]:
    return int(round(float(pair[0]))), int(round(float(pair[1])))


def pairwise_kpc(ego_seq: np.ndarray, other_seq: np.ndarray, k_horizon: int) -> int:
    ego_seq = hold_last_path(ego_seq, int(k_horizon))
    other_seq = hold_last_path(other_seq, int(k_horizon))
    steps = max(0, min(int(k_horizon), int(ego_seq.shape[0]) - 1, int(other_seq.shape[0]) - 1))
    if steps <= 0:
        return 0
    for t in range(1, steps + 1):
        e_prev = round_pair(ego_seq[t - 1])
        e_cur = round_pair(ego_seq[t])
        o_prev = round_pair(other_seq[t - 1])
        o_cur = round_pair(other_seq[t])
        if e_cur == o_cur or (e_prev == o_cur and e_cur == o_prev):
            return int(t)
    return 0


def _hold_and_round_path(path_abs: np.ndarray, k_horizon: int) -> np.ndarray:
    """hold_last_path + round → int32, returning shape (K+1, 2)."""
    seq = hold_last_path(path_abs, int(k_horizon))
    return np.round(seq).astype(np.int32)


def compute_min_kpc(paths_int: np.ndarray, k_horizon: int) -> np.ndarray:
    """Compute per-agent minimum KPC via fully vectorised NumPy broadcasting.

    For every ordered pair (i, j) the earliest conflict time is found; the
    per-agent minimum across all partners is returned.

    Two conflict types are detected at each step t = 1..K:
      * **Same-cell** — both agents occupy the same grid cell.
      * **Edge-swap** — agents cross the same edge in opposite directions
        (``prev[i] == cur[j]  AND  cur[i] == prev[j]``).

    Args:
        paths_int: int32 array of shape ``(n, K+1, 2)`` — pre-rounded via
            :func:`_hold_and_round_path`.
        k_horizon: number of look-ahead steps.

    Returns:
        int32 array of shape ``(n,)`` — per-agent minimum KPC
        (0 = no conflict within horizon).

    Complexity:
        Time:  ``O(n² · K)`` — one (n, n) broadcast per timestep, K steps.
        Space: ``O(n²)``     — the (n, n) conflict matrix and a few views.
    """
    n = int(paths_int.shape[0])
    if n <= 1:
        return np.zeros(n, dtype=np.int32)

    k = min(int(k_horizon), int(paths_int.shape[1]) - 1)
    if k <= 0:
        return np.zeros(n, dtype=np.int32)

    cur = paths_int[:, 1:]    # (n, k, 2) — positions at t = 1..k
    prev = paths_int[:, :-1]  # (n, k, 2) — positions at t = 0..k-1

    # Sentinel: k+1 means "no conflict found yet"
    min_kpc = np.full(n, k + 1, dtype=np.int32)

    for t in range(k):
        ct = cur[:, t]   # (n, 2)
        pt = prev[:, t]  # (n, 2)

        # --- Same-cell conflict: cur[i] == cur[j] ---
        # (n, 1, 2) vs (1, n, 2) → (n, n) boolean
        same = (ct[:, None, 0] == ct[None, :, 0]) & (ct[:, None, 1] == ct[None, :, 1])

        # --- Edge-swap: prev[i] == cur[j]  AND  cur[i] == prev[j] ---
        prev_eq_cur = (pt[:, None, 0] == ct[None, :, 0]) & (pt[:, None, 1] == ct[None, :, 1])
        cur_eq_prev = (ct[:, None, 0] == pt[None, :, 0]) & (ct[:, None, 1] == pt[None, :, 1])
        swap = prev_eq_cur & cur_eq_prev

        # Earliest conflict per pair: same-cell wins ties (lower val = earlier)
        conflict = same | swap
        val = t + 1

        # For each agent, reduce across all partners
        np.fill_diagonal(conflict, False)
        rows, cols = np.nonzero(conflict)
        if rows.size > 0:
            np.minimum.at(min_kpc, rows, val)

    # sentinel → 0 (no conflict)
    return np.where(min_kpc <= k, min_kpc, 0).astype(np.int32)


def ego_k_path_conflict(
    planner_paths: Mapping[str, Any],
    ego_agent: str,
    k_horizon: int,
    fallback_pos: tuple[int, int],
) -> int:
    ego_entry = planner_paths.get(ego_agent) if isinstance(planner_paths, Mapping) else None
    ego_seq = path_array_from_entry(ego_entry, fallback_pos=fallback_pos)
    best = 0
    for other_agent, other_entry in planner_paths.items():
        if str(other_agent) == str(ego_agent):
            continue
        other_seq = path_array_from_entry(other_entry, fallback_pos=fallback_pos)
        kpc = pairwise_kpc(ego_seq, other_seq, int(k_horizon))
        if kpc > 0 and (best == 0 or kpc < best):
            best = int(kpc)
    return int(best)


def build_union_msgs(
    planner_paths: Mapping[str, Any],
    ego_agent: str,
    current_pos: tuple[int, int],
    fov_radius_x: int,
    fov_radius_y: int,
    fov_w: int,
    fov_h: int,
    k_horizon: int,
) -> np.ndarray:
    msgs = np.zeros((int(k_horizon), int(fov_h), int(fov_w)), dtype=np.float32)
    ego_x, ego_y = int(current_pos[0]), int(current_pos[1])
    for other_agent, other_entry in planner_paths.items():
        if str(other_agent) == str(ego_agent):
            continue
        path_abs = hold_last_path(path_array_from_entry(other_entry, fallback_pos=(ego_x, ego_y)), int(k_horizon))
        other_cur = round_pair(path_abs[0])
        if abs(int(other_cur[0]) - ego_x) > int(fov_radius_x) or abs(int(other_cur[1]) - ego_y) > int(fov_radius_y):
            continue
        steps = min(int(k_horizon), int(path_abs.shape[0]) - 1)
        for t in range(steps):
            pos = round_pair(path_abs[t + 1])
            fx = int(fov_radius_x + pos[0] - ego_x)
            fy = int(fov_radius_y + pos[1] - ego_y)
            if 0 <= fx < int(fov_w) and 0 <= fy < int(fov_h):
                msgs[t, fy, fx] = 1.0
    return msgs


def copy_planner_paths_for_info(planner_paths: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for aid, entry in (planner_paths or {}).items():
        if not isinstance(entry, Mapping):
            continue
        out[str(aid)] = {
            "path_abs": np.asarray(entry.get("path_abs", []), dtype=np.float32).copy(),
            "alive": bool(entry.get("alive", True)),
            "has_path": bool(entry.get("has_path", False)),
        }
    return out
