from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

from SoftRHCR.modules.planner_features import (
    build_path_info,
    current_pos_from_obs,
    extract_shared_planner_paths,
    has_planner_path,
    hold_last_path,
    pairwise_kpc,
    path_array_from_entry,
    round_pair,
    _hold_and_round_path,
    compute_min_kpc,
)


def _agent_index(agent_id: str) -> int:
    return int(str(agent_id).split("_")[-1])


def _copy_core_obs(agent_obs: Mapping[str, Any]) -> Dict[str, Any]:
    # Shallow copy is usually sufficient since we only overwrite top-level keys like "msgs"
    out = dict(agent_obs)
    return out


@dataclass
class AdapterAux:
    k_path_conflict: int = 0
    committed_neighbors: int = 0
    interactive_neighbors: int = 0
    min_committed_kpc: int = 0
    min_interactive_dist: float = 0.0
    conflicted: bool = False
    invalid_action: bool = False
    task_completed: bool = False
    stagnant: bool = False
    progress_distance_prev: float = -1.0
    progress_distance_now: float = -1.0


class ObservationAdapterBase:
    def __init__(self, raw_obs_shape: Dict[str, Any], n_agents: int, config: Optional[Any] = None):
        self.raw_obs_shape = dict(raw_obs_shape)
        self.n_agents = int(n_agents)
        self.config = config
        self.k_horizon = int(self.raw_obs_shape.get("msgs", (0,))[0]) if isinstance(self.raw_obs_shape.get("msgs"), (tuple, list)) else 0
        self.map_width = int(self.raw_obs_shape.get("map_width", 1))
        self.map_height = int(self.raw_obs_shape.get("map_height", 1))
        fov_shape = tuple(self.raw_obs_shape.get("fov", (0, 0, 0)))
        self.fov_h = int(fov_shape[1]) if len(fov_shape) >= 3 else 0
        self.fov_w = int(fov_shape[2]) if len(fov_shape) >= 3 else 0
        self.fov_radius_y = self.fov_h // 2
        self.fov_radius_x = self.fov_w // 2
        self.policy_obs_shape = {
            "fov": tuple(raw_obs_shape["fov"]),
            "msgs": tuple(raw_obs_shape.get("msgs", (int(self.k_horizon), self.fov_h, self.fov_w))),
            "self_states": int(raw_obs_shape["self_states"]) + int(self.k_horizon) * 2 + 1,
        }
        self.last_aux: Dict[str, Dict[str, Any]] = {}

    def adapt(
        self,
        obs: Dict[str, Any],
        adapter_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not isinstance(obs, dict):
            self.last_aux = {}
            return obs
        out = {aid: _copy_core_obs(agent_obs) for aid, agent_obs in obs.items() if isinstance(agent_obs, Mapping)}
        self.last_aux = {aid: AdapterAux().__dict__.copy() for aid in out.keys()}
        return out

    def reset_episode(self) -> None:
        self.last_aux = {}

    def reset_done_agents(self, dones: Optional[Dict[str, Any]]) -> None:
        if not isinstance(dones, dict):
            return
        for aid, done in dones.items():
            if bool(done):
                self.last_aux.pop(str(aid), None)

    @staticmethod
    def _flat_dim(obj: Any) -> int:
        if isinstance(obj, np.ndarray):
            return int(np.prod(obj.shape))
        if isinstance(obj, dict):
            return sum(ObservationAdapterBase._flat_dim(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(ObservationAdapterBase._flat_dim(v) for v in obj)
        if isinstance(obj, (int, float, np.floating, np.integer)):
            return 1
        return 0

    def compute_adapted_obs_shape(self, n_agents: int) -> Dict[str, Any]:
        try:
            return self._measure_via_dry_run(int(n_agents))
        except Exception:
            return dict(self.policy_obs_shape)

    def _measure_via_dry_run(self, n_agents: int) -> Dict[str, Any]:
        fov_shape = tuple(self.raw_obs_shape.get("fov", (5, 1, 1)))
        ss_raw_dim = int(self.raw_obs_shape.get("self_states", 0))
        num_agents = max(int(n_agents), 1)

        dummy_obs: Dict[str, Any] = {}
        for i in range(min(num_agents, 2)):
            aid = f"__dryrun_a{i}__"
            dummy_obs[aid] = {
                "self_states": {
                    "position": np.zeros(2, dtype=np.float32),
                    "__dryrun_rest__": np.zeros(max(ss_raw_dim - 2, 0), dtype=np.float32),
                },
                "fov": np.zeros(fov_shape, dtype=np.float32),
            }

        dummy_state: Dict[str, Any] = {
            "env_info": {
                "__dryrun__": {
                    "planner_paths": {},
                    "action_mask": None,
                }
            }
        }

        adapted = self.adapt(dummy_obs, adapter_state=dummy_state)
        if not isinstance(adapted, dict) or len(adapted) == 0:
            raise RuntimeError("adapt() returned empty or non-dict result")

        sample = next(iter(adapted.values()))
        if not isinstance(sample, dict):
            raise RuntimeError("adapt() returned agent observation that is not dict")

        fov_out = sample.get("fov")
        msgs_out = sample.get("msgs")
        ss_out = sample.get("self_states", {})

        return {
            "fov": tuple(fov_out.shape) if isinstance(fov_out, np.ndarray) else fov_shape,
            "msgs": tuple(msgs_out.shape) if isinstance(msgs_out, np.ndarray) else (int(self.k_horizon), self.fov_h, self.fov_w),
            "self_states": self._flat_dim(ss_out),
        }

    def _zero_msgs(self, channels: int) -> np.ndarray:
        return np.zeros((int(channels), self.fov_h, self.fov_w), dtype=np.float32)

    def _build_step_aux(self, env_info: Optional[Dict[str, Any]], aid: str) -> Dict[str, Any]:
        item = env_info.get(aid) if isinstance(env_info, dict) else None
        if not isinstance(item, Mapping):
            return {}
        progress_distance_prev = item.get("progress_distance_prev", -1.0)
        progress_distance_now = item.get("progress_distance_now", -1.0)
        try:
            progress_distance_prev = float(progress_distance_prev)
        except Exception:
            progress_distance_prev = -1.0
        try:
            progress_distance_now = float(progress_distance_now)
        except Exception:
            progress_distance_now = -1.0
        return {
            "conflicted": bool(item.get("conflicted", False)),
            "invalid_action": bool(item.get("invalid_action", False)),
            "task_completed": bool(item.get("task_completed", False)),
            "progress_distance_prev": progress_distance_prev,
            "progress_distance_now": progress_distance_now,
            "stagnant": bool(item.get("stagnant", False)),
        }

    def _project_future_occupancy(
        self,
        ego_pos: tuple[int, int],
        other_seq: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        msgs = np.zeros((int(self.k_horizon), self.fov_h, self.fov_w), dtype=np.float32)
        if int(self.k_horizon) <= 0 or not isinstance(other_seq, np.ndarray) or other_seq.ndim != 2 or other_seq.shape[1] != 2:
            return msgs, False

        ego_x, ego_y = int(ego_pos[0]), int(ego_pos[1])
        other_cur = round_pair(other_seq[0])
        in_fov_now = (
            abs(int(other_cur[0]) - ego_x) <= int(self.fov_radius_x)
            and abs(int(other_cur[1]) - ego_y) <= int(self.fov_radius_y)
        )
        if not in_fov_now:
            return msgs, False

        other_seq = hold_last_path(other_seq, int(self.k_horizon))
        steps = min(int(self.k_horizon), int(other_seq.shape[0]) - 1)
        if steps <= 0:
            return msgs, True

        pos_seq = np.round(other_seq[1 : steps + 1]).astype(np.int32)
        fx_seq = int(self.fov_radius_x) + pos_seq[:, 0] - ego_x
        fy_seq = int(self.fov_radius_y) + pos_seq[:, 1] - ego_y
        valid_mask = (fx_seq >= 0) & (fx_seq < int(self.fov_w)) & (fy_seq >= 0) & (fy_seq < int(self.fov_h))
        valid_t = np.where(valid_mask)[0]
        if valid_t.size > 0:
            msgs[valid_t, fy_seq[valid_t], fx_seq[valid_t]] = 1.0
        return msgs, True

    def _build_planner_batch_context(
        self,
        obs: Dict[str, Any],
        adapter_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(obs, dict) or self.k_horizon <= 0:
            return None
        env_info = adapter_state.get("env_info") if isinstance(adapter_state, dict) else None
        planner_paths = extract_shared_planner_paths(env_info if isinstance(env_info, dict) else None)
        if not isinstance(planner_paths, Mapping) or len(planner_paths) == 0:
            return None

        agent_ids = [str(aid) for aid, agent_obs in obs.items() if isinstance(agent_obs, Mapping)]
        current_positions: Dict[str, tuple[int, int]] = {}
        entries: Dict[str, Optional[Mapping[str, Any]]] = {}
        seqs: Dict[str, np.ndarray] = {}
        other_current_pos: Dict[str, tuple[int, int]] = {}

        for aid in agent_ids:
            agent_obs = obs.get(aid)
            if not isinstance(agent_obs, Mapping):
                continue
            entry = planner_paths.get(aid) if isinstance(planner_paths, Mapping) else None
            entries[aid] = entry if isinstance(entry, Mapping) else None
            # Prefer path[0] as current position when planner path is available;
            # fall back to obs de-normalization otherwise.
            if isinstance(entries[aid], Mapping) and bool(entries[aid].get("has_path", False)):
                seq_tmp = path_array_from_entry(entries[aid], fallback_pos=(0, 0))
                cur_pos = (int(round(float(seq_tmp[0][0]))), int(round(float(seq_tmp[0][1]))))
            else:
                cur_pos = current_pos_from_obs(agent_obs, self.map_width, self.map_height)
            current_positions[aid] = cur_pos
            seqs[aid] = path_array_from_entry(entries[aid], fallback_pos=cur_pos)

        for other_aid, other_entry in planner_paths.items():
            other_name = str(other_aid)
            entry_map = other_entry if isinstance(other_entry, Mapping) else None
            other_seq = path_array_from_entry(entry_map, fallback_pos=(0, 0))
            entries.setdefault(other_name, entry_map)
            seqs.setdefault(other_name, other_seq)
            other_current_pos[other_name] = round_pair(other_seq[0])

        relevant_names = [
            str(aid)
            for aid, entry in planner_paths.items()
            if str(aid) in seqs and bool((entry or {}).get("alive", True))
        ]
        # Pre-compute rounded int32 paths (avoids repeated hold_last_path + round
        # in the per-agent loops and in the KPC precomputation).
        seqs_int: Dict[str, np.ndarray] = {}
        seqs_int_list: list[np.ndarray] = []
        agent_order: list[str] = []
        for name in relevant_names:
            raw = seqs.get(name)
            if raw is not None:
                rounded = _hold_and_round_path(raw, int(self.k_horizon))
                seqs_int[name] = rounded
                agent_order.append(name)
                seqs_int_list.append(rounded)

        # Compute per-agent minimum KPC via spatial hashing
        # O(n * K) average case instead of O(n² * K) for brute-force pair iteration.
        agent_min_kpc: Dict[str, int] = {aid: 0 for aid in relevant_names}
        if len(seqs_int_list) > 1:
            try:
                stacked = np.stack(seqs_int_list, axis=0)  # (n, K+1, 2)
                min_kpc_arr = compute_min_kpc(stacked, int(self.k_horizon))
                for idx, name in enumerate(agent_order):
                    agent_min_kpc[name] = int(min_kpc_arr[idx])
            except Exception:
                pass  # fallback: all stay 0

        return {
            "agent_ids": agent_ids,
            "planner_paths": planner_paths,
            "current_positions": current_positions,
            "entries": entries,
            "seqs": seqs,
            "seqs_int": seqs_int,
            "agent_order": agent_order,
            "agent_min_kpc": agent_min_kpc,
            "other_current_pos": other_current_pos,
        }

    def _compute_default_batch_features(self, batch_ctx: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        if not isinstance(batch_ctx, dict):
            return out
        planner_paths = batch_ctx["planner_paths"]
        current_positions = batch_ctx["current_positions"]
        entries = batch_ctx["entries"]
        seqs = batch_ctx["seqs"]
        seqs_int = batch_ctx["seqs_int"]
        agent_min_kpc = batch_ctx["agent_min_kpc"]

        for aid in batch_ctx["agent_ids"]:
            cur_pos = current_positions.get(aid, (0, 0))
            ego_entry = entries.get(aid)
            ego_seq = seqs.get(aid)
            if ego_seq is None:
                out[aid] = {
                    "path_info": np.zeros((int(self.k_horizon) * 2 + 1,), dtype=np.float32),
                    "msgs": self._zero_msgs(self.k_horizon),
                    "aux": AdapterAux().__dict__.copy(),
                }
                continue

            path_info = build_path_info(
                ego_seq,
                current_pos=cur_pos,
                k_horizon=int(self.k_horizon),
                map_width=int(self.map_width),
                map_height=int(self.map_height),
                has_path=has_planner_path(ego_entry),
            )
            msgs = self._zero_msgs(self.k_horizon)
            best_kpc = int(agent_min_kpc.get(aid, 0))
            ego_x, ego_y = int(cur_pos[0]), int(cur_pos[1])

            for other_name in planner_paths.keys():
                other_name = str(other_name)
                if other_name == aid:
                    continue
                other_seq_int = seqs_int.get(other_name)
                if other_seq_int is not None:
                    # Use pre-rounded int32 path — avoids redundant round() calls
                    other_cur = (int(other_seq_int[0, 0]), int(other_seq_int[0, 1]))
                    in_fov_now = (
                        abs(other_cur[0] - ego_x) <= int(self.fov_radius_x)
                        and abs(other_cur[1] - ego_y) <= int(self.fov_radius_y)
                    )
                    if not in_fov_now:
                        continue
                    # Project pre-rounded path onto FoV msgs
                    steps = min(int(self.k_horizon), int(other_seq_int.shape[0]) - 1)
                    if steps > 0:
                        pos_seq = other_seq_int[1: steps + 1]  # already int32
                        fx_seq = int(self.fov_radius_x) + pos_seq[:, 0] - ego_x
                        fy_seq = int(self.fov_radius_y) + pos_seq[:, 1] - ego_y
                        valid_mask = (fx_seq >= 0) & (fx_seq < int(self.fov_w)) & (fy_seq >= 0) & (fy_seq < int(self.fov_h))
                        valid_t = np.where(valid_mask)[0]
                        if valid_t.size > 0:
                            channel_msgs = np.zeros((int(self.k_horizon), self.fov_h, self.fov_w), dtype=np.float32)
                            channel_msgs[valid_t, fy_seq[valid_t], fx_seq[valid_t]] = 1.0
                            msgs = np.maximum(msgs, channel_msgs)

            out[aid] = {
                "path_info": path_info,
                "msgs": msgs,
                "aux": AdapterAux(k_path_conflict=int(best_kpc)).__dict__.copy(),
            }
        return out

class DefaultObservationAdapter(ObservationAdapterBase):
    def adapt(
        self,
        obs: Dict[str, Any],
        adapter_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not isinstance(obs, dict):
            self.last_aux = {}
            return obs
        env_info = adapter_state.get("env_info") if isinstance(adapter_state, dict) else None
        batch_ctx = self._build_planner_batch_context(obs, adapter_state=adapter_state)
        planner_feats = self._compute_default_batch_features(batch_ctx)
        out: Dict[str, Any] = {}
        aux_out: Dict[str, Dict[str, Any]] = {}
        for aid, agent_obs in obs.items():
            if not isinstance(agent_obs, Mapping):
                continue
            core_obs = _copy_core_obs(agent_obs)
            ss = dict(core_obs.get("self_states") or {})
            feat = planner_feats.get(aid)
            if isinstance(feat, dict):
                ss["path_info"] = feat["path_info"]
                core_obs["msgs"] = feat["msgs"]
                aux_payload = dict(feat["aux"])
            else:
                ss["path_info"] = np.zeros((int(self.k_horizon) * 2 + 1,), dtype=np.float32)
                core_obs["msgs"] = self._zero_msgs(self.k_horizon)
                aux_payload = AdapterAux().__dict__.copy()
            aux_payload.update(self._build_step_aux(env_info, str(aid)))
            aux_out[aid] = aux_payload
            core_obs["self_states"] = ss
            out[aid] = core_obs
        self.last_aux = aux_out
        return out


class SoftRHCRObservationAdapter(ObservationAdapterBase):
    def __init__(self, raw_obs_shape: Dict[str, Any], n_agents: int, config: Optional[Any] = None):
        super().__init__(raw_obs_shape, n_agents, config=config)
        msgs_mode = str(getattr(config, "soft_rhcr_msgs_mode", "dual") or "dual").strip().lower()
        if msgs_mode not in ("single", "dual"):
            msgs_mode = "dual"
        self.msgs_mode = msgs_mode
        self.msg_channels = int(self.k_horizon * 2) if self.msgs_mode == "dual" else int(self.k_horizon)
        self.policy_obs_shape = {
            "fov": tuple(raw_obs_shape["fov"]),
            "msgs": (int(self.msg_channels), self.fov_h, self.fov_w),
            "self_states": int(raw_obs_shape["self_states"]) + int(self.k_horizon) * 2 + 1,
        }

    @staticmethod
    def _round_pair(pair: np.ndarray) -> tuple[int, int]:
        return int(round(float(pair[0]))), int(round(float(pair[1])))

    @staticmethod
    def _grid_dist(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))

    def compute_gate_context(
        self,
        obs: Dict[str, Any],
        adapter_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute gate-decision context (KPC + planner path context).

        This is a lightweight step that does NOT build msgs — it only computes
        per-agent minimum KPC and assembles the internal batch_ctx that will be
        reused by build_policy_observation().

        Returns
        -------
        Dict with keys:
            agent_ids      : List[str]  — sorted agent id list
            agent_min_kpc  : Dict[str, int]  — per-agent minimum KPC
            batch_ctx      : Optional[Dict]  — internal context for build_policy_observation
            aux_base       : Dict[str, Dict]  — base aux info per agent
                             (k_path_conflict from agent_min_kpc +
                              task_completed / conflicted etc. from env_info)
        """
        env_info = adapter_state.get("env_info") if isinstance(adapter_state, dict) else None
        batch_ctx = self._build_planner_batch_context(obs, adapter_state=adapter_state)

        agent_ids: list[str] = []
        agent_min_kpc: Dict[str, int] = {}
        aux_base: Dict[str, Dict[str, Any]] = {}

        if isinstance(obs, dict):
            agent_ids = [str(aid) for aid, v in obs.items() if isinstance(v, Mapping)]

        if isinstance(batch_ctx, dict):
            agent_min_kpc = dict(batch_ctx.get("agent_min_kpc", {}))

        for aid in agent_ids:
            step_aux = self._build_step_aux(env_info, aid)
            kpc = int(agent_min_kpc.get(aid, 0))
            base: Dict[str, Any] = {"k_path_conflict": kpc}
            base.update(step_aux)
            aux_base[aid] = base

        return {
            "agent_ids": agent_ids,
            "agent_min_kpc": agent_min_kpc,
            "batch_ctx": batch_ctx,
            "aux_base": aux_base,
        }

    def build_policy_observation(
        self,
        obs: Dict[str, Any],
        gate_ctx: Dict[str, Any],
        committed_agents: Dict[str, bool],
    ) -> Dict[str, Any]:
        """Build policy-network input based on precise committed state.

        Parameters
        ----------
        obs              : raw observation dict
        gate_ctx         : return value of compute_gate_context()
        committed_agents : Dict[str, bool] — per-agent committed flag after
                           gate decision (use_fp_dict from select_action)

        Returns
        -------
        Dict[str, Any] — same format as adapt(): one dict per agent with
        keys fov, msgs, self_states.
        """
        if not isinstance(obs, dict):
            self.last_aux = {}
            return obs

        batch_ctx: Optional[Dict[str, Any]] = gate_ctx.get("batch_ctx") if isinstance(gate_ctx, dict) else None
        agent_min_kpc: Dict[str, int] = gate_ctx.get("agent_min_kpc", {}) if isinstance(gate_ctx, dict) else {}
        aux_base: Dict[str, Dict[str, Any]] = gate_ctx.get("aux_base", {}) if isinstance(gate_ctx, dict) else {}
        env_info = None  # env_info already consumed into aux_base during compute_gate_context

        out: Dict[str, Any] = {}
        aux_out: Dict[str, Dict[str, Any]] = {}

        for aid, agent_obs in obs.items():
            if not isinstance(agent_obs, Mapping):
                continue
            base_obs = _copy_core_obs(agent_obs)
            ss = dict(base_obs.get("self_states") or {})
            if not isinstance(batch_ctx, dict):
                ss["path_info"] = np.zeros((int(self.k_horizon) * 2 + 1,), dtype=np.float32)
                base_obs["self_states"] = ss
                base_obs["msgs"] = self._zero_msgs(self.msg_channels)
                out[aid] = base_obs
                aux_payload = AdapterAux().__dict__.copy()
                aux_payload.update(aux_base.get(str(aid), {}))
                aux_out[aid] = aux_payload
                continue

            planner_paths = batch_ctx["planner_paths"]
            seqs_int = batch_ctx["seqs_int"]
            current_pos = batch_ctx["current_positions"].get(
                aid,
                current_pos_from_obs(agent_obs, self.map_width, self.map_height),
            )
            ego_entry = batch_ctx["entries"].get(aid)
            ego_seq = batch_ctx["seqs"].get(aid)
            if ego_seq is None:
                ss["path_info"] = np.zeros((int(self.k_horizon) * 2 + 1,), dtype=np.float32)
                base_obs["self_states"] = ss
                base_obs["msgs"] = self._zero_msgs(self.msg_channels)
                out[aid] = base_obs
                aux_payload = AdapterAux().__dict__.copy()
                aux_payload.update(aux_base.get(str(aid), {}))
                aux_out[aid] = aux_payload
                continue

            ss["path_info"] = build_path_info(
                ego_seq,
                current_pos=current_pos,
                k_horizon=int(self.k_horizon),
                map_width=int(self.map_width),
                map_height=int(self.map_height),
                has_path=has_planner_path(ego_entry),
            )
            base_obs["self_states"] = ss
            ego_pos = (int(current_pos[0]), int(current_pos[1]))
            committed_msgs = np.zeros((self.k_horizon, self.fov_h, self.fov_w), dtype=np.float32)
            interactive_msgs = np.zeros((self.k_horizon, self.fov_h, self.fov_w), dtype=np.float32)

            committed_neighbors = 0
            interactive_neighbors = 0
            ego_kpc = int(agent_min_kpc.get(aid, 0))
            min_committed_kpc = 0
            min_interactive_dist = 0.0

            ego_seq_int = seqs_int.get(str(aid))
            for other_name in planner_paths.keys():
                other_name = str(other_name)
                if other_name == aid:
                    continue
                other_seq_int = seqs_int.get(other_name)
                if other_seq_int is None:
                    continue
                other_cur = (int(other_seq_int[0, 0]), int(other_seq_int[0, 1]))
                in_fov = (
                    abs(other_cur[0] - ego_pos[0]) <= int(self.fov_radius_x)
                    and abs(other_cur[1] - ego_pos[1]) <= int(self.fov_radius_y)
                )
                if not in_fov:
                    continue

                # Use precise committed state from gate decision
                is_committed = bool(committed_agents.get(other_name, False))

                # Compute KPC on-demand for committed agents only (small subset)
                if is_committed and ego_seq_int is not None:
                    current_kpc = int(pairwise_kpc(
                        ego_seq_int.astype(np.float32),
                        other_seq_int.astype(np.float32),
                        int(self.k_horizon),
                    ))
                    if current_kpc > 0 and (min_committed_kpc == 0 or current_kpc < min_committed_kpc):
                        min_committed_kpc = int(current_kpc)

                # Project pre-rounded path onto FoV msgs
                steps = min(int(self.k_horizon), int(other_seq_int.shape[0]) - 1)
                if steps > 0:
                    pos_seq = other_seq_int[1: steps + 1]
                    fx_seq = int(self.fov_radius_x) + pos_seq[:, 0] - ego_pos[0]
                    fy_seq = int(self.fov_radius_y) + pos_seq[:, 1] - ego_pos[1]
                    valid_mask = (fx_seq >= 0) & (fx_seq < int(self.fov_w)) & (fy_seq >= 0) & (fy_seq < int(self.fov_h))
                    valid_t = np.where(valid_mask)[0]
                    if valid_t.size > 0:
                        channel_msgs = np.zeros((int(self.k_horizon), self.fov_h, self.fov_w), dtype=np.float32)
                        channel_msgs[valid_t, fy_seq[valid_t], fx_seq[valid_t]] = 1.0

                        if is_committed:
                            committed_neighbors += 1
                            committed_msgs = np.maximum(committed_msgs, channel_msgs)
                        else:
                            interactive_neighbors += 1
                            interactive_msgs = np.maximum(interactive_msgs, channel_msgs)
                            cur_dist = float(self._grid_dist(ego_pos, other_cur))
                            if min_interactive_dist <= 0.0 or cur_dist < min_interactive_dist:
                                min_interactive_dist = cur_dist

            if self.msgs_mode == "dual":
                msgs = np.concatenate([committed_msgs, interactive_msgs], axis=0)
            else:
                msgs = np.maximum(committed_msgs, interactive_msgs)
            base_obs["msgs"] = msgs
            out[aid] = base_obs
            aux_payload = AdapterAux(
                k_path_conflict=int(ego_kpc),
                committed_neighbors=int(committed_neighbors),
                interactive_neighbors=int(interactive_neighbors),
                min_committed_kpc=int(min_committed_kpc),
                min_interactive_dist=float(min_interactive_dist),
            ).__dict__.copy()
            # Merge base aux (env_info fields) gathered in compute_gate_context
            for k, v in aux_base.get(str(aid), {}).items():
                if k not in ("k_path_conflict",):  # k_path_conflict already set above
                    aux_payload[k] = v
            aux_out[aid] = aux_payload

        self.last_aux = aux_out
        return out

    def adapt(
        self,
        obs: Dict[str, Any],
        adapter_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compatibility entry point: compute_gate_context + build_policy_observation.

        The committed state is derived from adapter_state["planner_commit_remaining"]
        (legacy call convention).
        """
        if not isinstance(obs, dict):
            self.last_aux = {}
            return obs

        # Derive committed_agents from planner_commit_remaining (legacy path)
        committed_agents: Dict[str, bool] = {}
        if isinstance(adapter_state, dict):
            raw_commit = adapter_state.get("planner_commit_remaining")
            if isinstance(raw_commit, dict):
                committed_agents = {str(k): (int(v) > 0) for k, v in raw_commit.items()}

        gate_ctx = self.compute_gate_context(obs, adapter_state=adapter_state)
        return self.build_policy_observation(obs, gate_ctx, committed_agents)
