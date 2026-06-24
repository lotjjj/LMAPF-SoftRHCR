from __future__ import annotations

import copy
import multiprocessing as mp
import queue
import re
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Optional

import numpy as np
from tqdm import tqdm

from LMAPFEnv import WarehouseEnv
from LMAPFEnv.algorithms.path_planners import PlannerPolicy
from SoftRHCR.config.configBase import (
    EnvConfig,
    SoftRHCRConfig,
    SoftRHCRMAPPOConfig,
    algo_config_from_dict,
)
from SoftRHCR.config.config_loader import load_and_build_config
from SoftRHCR.scripts.cli_review import confirm_resolved_config
from SoftRHCR.config.registry import (
    agent_select_action_requires_info,
    create_agent as registry_create_agent,
)
from SoftRHCR.modules.model_io import (
    create_and_load_agent_from_model_spec,
    load_agent_state,
    resolve_model_load_spec,
)
# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TQDM_PROGRESS_RE = re.compile(r"\d{1,3}%\|")
BAR_PROGRESS_RE = re.compile(r"^\[[=\-]{8,}\]\s+\d+\.\d+%")

# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------
class EvaluationStopped(Exception):
    pass


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", str(text))


def looks_like_progress_line(text: str) -> bool:
    line = strip_ansi(text).strip()
    if line == "":
        return False
    if TQDM_PROGRESS_RE.search(line) is not None:
        return True
    if BAR_PROGRESS_RE.search(line) is not None:
        return True
    if "steps/s" in line or "it/s" in line:
        return True
    if " ETA " in line and "%" in line:
        return True
    if "steps " in line and "episode " in line and "%" in line:
        return True
    return False


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


class _CliEvaluationProgress:
    def __init__(self, total_episodes: int) -> None:
        self.total_episodes = max(1, int(total_episodes))
        self.completed = 0
        self._bar: Optional[Any] = None

    def __enter__(self) -> "_CliEvaluationProgress":
        self._bar = tqdm(
            total=self.total_episodes,
            desc="Evaluate",
            unit="ep",
            dynamic_ncols=True,
            mininterval=0.2,
            maxinterval=1.0,
            leave=True,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def update_state(self, payload: dict[str, Any]) -> None:
        if self._bar is None:
            return
        completed = max(0, int(payload.get("completed_episodes", 0) or 0))
        if completed > self.completed:
            self._bar.update(completed - self.completed)
            self.completed = completed
        active_workers = max(0, int(payload.get("active_workers", 0) or 0))
        eval_threads = max(1, int(payload.get("eval_threads", 1) or 1))
        elapsed = float(payload.get("elapsed_sec", 0.0) or 0.0)
        step_rate = float(payload.get("steps_per_sec", 0.0) or 0.0)
        postfix = [f"thr={active_workers}/{eval_threads}"]
        if step_rate > 0.0:
            postfix.append(f"{step_rate:.1f}step/s")
        if elapsed > 0.0:
            postfix.append(f"{elapsed:.0f}s")
        self._bar.set_postfix_str(" ".join(postfix), refresh=False)

    def finalize(self, metrics: dict[str, Any]) -> None:
        if self._bar is None:
            return
        if self.completed < self.total_episodes:
            self._bar.update(self.total_episodes - self.completed)
            self.completed = self.total_episodes
        postfix: list[str] = []
        if "task_completion_mean" in metrics:
            postfix.append(f"task={float(metrics['task_completion_mean']):.1f}")
        elif "tasks_completed_mean" in metrics:
            postfix.append(f"task={float(metrics['tasks_completed_mean']):.1f}")
        if "conflict_times_mean" in metrics:
            postfix.append(f"conf={float(metrics['conflict_times_mean']):.1f}")
        if "elapsed_sec" in metrics:
            postfix.append(f"{float(metrics['elapsed_sec']):.0f}s")
        if postfix:
            self._bar.set_postfix_str(" ".join(postfix), refresh=True)


def infer_step_from_name(path: Path) -> Optional[int]:
    stem = path.stem
    marker = "_step"
    if marker not in stem:
        return None
    try:
        return int(stem.split(marker)[-1])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Environment / agent helpers
# ---------------------------------------------------------------------------
def extract_obs_info(env: WarehouseEnv) -> dict[str, Any]:
    base_env = getattr(env, "_env", env)
    agent_id = env.possible_agents[0]
    obs_space = env.observation_spaces[agent_id]
    self_state_dim = 0
    if hasattr(obs_space, "spaces") and "self_states" in obs_space.spaces:
        for sub in obs_space.spaces["self_states"].spaces.values():
            if hasattr(sub, "shape"):
                self_state_dim += int(np.prod(sub.shape))
    fov_shape = obs_space.spaces["fov"].shape
    if hasattr(obs_space, "spaces") and "msgs" in obs_space.spaces:
        msg_shape = obs_space.spaces["msgs"].shape
    else:
        msg_shape = (
            int(getattr(base_env, "kstep_conflict_check", 0)),
            int(fov_shape[1]),
            int(fov_shape[2]),
        )
    return {
        "fov": fov_shape,
        "msgs": msg_shape,
        "self_states": self_state_dim,
        "map_width": int(getattr(base_env, "width", 1)),
        "map_height": int(getattr(base_env, "height", 1)),
    }


def create_agent(algo_cfg: Any, obs_info: dict[str, Any], action_dim: int, n_agents: int):
    return registry_create_agent(algo_cfg, obs_info, action_dim, n_agents)


class PlannerEvalAgent:
    def __init__(self) -> None:
        self._policy: Optional[PlannerPolicy] = None

    def bind_env(self, env: WarehouseEnv) -> None:
        planner = getattr(env, "path_planner", None)
        if planner is None:
            raise RuntimeError("The environment has no path_planner enabled; cannot run the planner evaluation.")
        self._policy = PlannerPolicy(planner)

    def set_training_mode(self, mode: bool = True) -> None:
        return None

    def select_env_actions(
        self,
        env: WarehouseEnv,
        obs: dict[str, Any],
        info: Optional[dict[str, Any]],
        action_masks: Optional[dict[str, Any]],
    ) -> dict[str, int]:
        if self._policy is None:
            self.bind_env(env)
        return dict(self._policy.select_actions(env.agvs, env.agents))


def load_agent_for_evaluation(agent: Any, model_path: str) -> None:
    load_agent_state(agent, model_path, evaluation=True)


def get_action_masks(
    env: WarehouseEnv, obs: dict[str, Any], info: Optional[dict[str, Any]]
) -> Optional[dict[str, Any]]:
    if isinstance(info, dict):
        masks: dict[str, Any] = {}
        for aid in obs.keys():
            item = info.get(aid) if isinstance(info.get(aid), dict) else None
            if item is not None and "action_mask" in item:
                masks[aid] = item["action_mask"]
        if masks:
            return masks
    if hasattr(env, "action_mask"):
        try:
            return {aid: env.action_mask(aid) for aid in obs.keys()}
        except Exception:
            return None
    return None


def select_actions(
    env: WarehouseEnv,
    agent: Any,
    obs: dict[str, Any],
    info: Optional[dict[str, Any]],
    action_masks: Optional[dict[str, Any]],
) -> dict[str, int]:
    if hasattr(agent, "select_env_actions"):
        return dict(agent.select_env_actions(env, obs, info, action_masks))
    if hasattr(agent, "set_observation_context"):
        agent.set_observation_context(info)
    kwargs = {"evaluation": True, "action_masks": action_masks}
    if agent_select_action_requires_info(agent):
        result = agent.select_action(obs, info=info, **kwargs)
    else:
        result = agent.select_action(obs, **kwargs)
    if isinstance(result, tuple):
        return dict(result[0])
    return dict(result)


# ---------------------------------------------------------------------------
# Seed / worker helpers
# ---------------------------------------------------------------------------
def build_eval_seed_list(env_cfg: EnvConfig) -> list[int]:
    eval_episodes = max(1, int(env_cfg.eval_episodes))
    base_seed = int(env_cfg.eval.seed)
    return [base_seed + episode for episode in range(eval_episodes)]


def split_eval_seeds_evenly(
    seeds: list[int], eval_threads: int
) -> list[list[tuple[int, int]]]:
    if not seeds:
        return []
    worker_count = max(1, min(int(eval_threads), len(seeds)))
    groups: list[list[tuple[int, int]]] = [[] for _ in range(worker_count)]
    for episode_index, seed in enumerate(seeds):
        groups[episode_index % worker_count].append((episode_index, int(seed)))
    return [group for group in groups if group]


def format_seed_group_compact(group: list[tuple[int, int]]) -> str:
    if not group:
        return "-"
    seeds = [int(seed) for _, seed in group]
    if len(seeds) <= 4:
        return ",".join(str(seed) for seed in seeds)
    return f"{seeds[0]},{seeds[1]},...,{seeds[-1]} ({len(seeds)})"


def summarize_seed_groups(groups: list[list[tuple[int, int]]]) -> list[str]:
    summary: list[str] = []
    for worker_index, group in enumerate(groups, start=1):
        summary.append(
            f"T{worker_index}[{len(group)}]: {format_seed_group_compact(group)}"
        )
    return summary


def _should_stop_evaluation(*events: Optional[threading.Event]) -> bool:
    for event in events:
        if event is not None and event.is_set():
            return True
    return False


# ---------------------------------------------------------------------------
# Core evaluation helpers
# ---------------------------------------------------------------------------
def _run_eval_seed_group(
    agent_factory,
    env_cfg: EnvConfig,
    seed_group: list[tuple[int, int]],
    worker_index: int,
    worker_count: int,
    total_episodes: int,
    render_interval_s: float,
    stop_event: Optional[Any],
    cancel_event: Optional[Any],
    progress_cb,
    episode_done_cb,
    task_limit: int = 0,
) -> list[dict[str, Any]]:
    # task_limit mode: use a generous step cap but do not mutate the original env_cfg
    if task_limit > 0:
        env_cfg = copy.deepcopy(env_cfg)
        env_cfg.eval.max_episode_steps = 10000
    env = WarehouseEnv(**env_cfg.get_env_args(mode="eval"))
    n_agents = len(env.possible_agents)
    try:
        agent = agent_factory()
        if hasattr(agent, "bind_env"):
            agent.bind_env(env)
        if hasattr(agent, "set_training_mode"):
            agent.set_training_mode(False)
        results: list[dict[str, Any]] = []
        for local_episode_index, (episode_index, seed) in enumerate(
            seed_group, start=1
        ):
            if _should_stop_evaluation(stop_event, cancel_event):
                raise EvaluationStopped("Evaluation terminated by the user.")
            obs, info = env.reset(seed=int(seed))
            if hasattr(agent, "reset_episode"):
                agent.reset_episode()

            ep_return = 0.0
            step_reward_sum = 0.0
            step_conflict_times = 0.0
            step_invalid_sum = 0.0
            task_completed_sum = 0.0
            episode_sum_of_costs = 0
            step_count = 0
            use_fp_step_sum = 0.0

            while True:
                if _should_stop_evaluation(stop_event, cancel_event):
                    raise EvaluationStopped("Evaluation terminated by the user.")
                action_masks = get_action_masks(env, obs, info)
                actions = select_actions(env, agent, obs, info, action_masks)
                next_obs, rewards, terminations, truncations, next_info = env.step(
                    actions
                )
                dones = {
                    aid: bool(terminations[aid] or truncations[aid])
                    for aid in terminations.keys()
                }
                if hasattr(agent, "reset_done_agents"):
                    agent.reset_done_agents(dones)
                if (
                    env_cfg.eval.render_mode is not None
                    and render_interval_s > 0.0
                ):
                    time.sleep(render_interval_s)

                keys = list(rewards.keys())
                ep_return += float(sum(rewards.values()))
                step_reward_sum += (
                    float(
                        np.mean(
                            np.asarray(list(rewards.values()), dtype=np.float32)
                        )
                    )
                    if rewards
                    else 0.0
                )

                # non-stay actions (action 4 = stay)
                non_stay_actions = sum(1 for a in actions.values() if a != 4)
                episode_sum_of_costs += non_stay_actions

                if isinstance(next_info, dict) and keys:
                    conflict_count = float(
                        sum(
                            1.0
                            for aid in keys
                            if bool(
                                (next_info.get(aid) or {}).get(
                                    "conflicted", False
                                )
                            )
                        )
                    )
                    step_conflict_times += conflict_count
                    step_invalid_sum += float(
                        np.mean(
                            [
                                1.0
                                if bool(
                                    (next_info.get(aid) or {}).get(
                                        "invalid_action", False
                                    )
                                )
                                else 0.0
                                for aid in keys
                            ]
                        )
                    )
                    task_completed_sum += float(
                        sum(
                            1.0
                            for aid in keys
                            if bool(
                                (next_info.get(aid) or {}).get(
                                    "task_completed", False
                                )
                            )
                        )
                    )

                step_count += 1
                obs, info = next_obs, next_info

                if all(dones.values()):
                    break

                if task_limit > 0 and task_completed_sum >= task_limit:
                    break  # reached target, terminate this episode early

            mean_step_reward = step_reward_sum / float(max(1, step_count))
            invalid_action_rate = step_invalid_sum / float(max(1, step_count))
            use_fp_rate = use_fp_step_sum / float(max(1, step_count))
            result = {
                "episode_index": int(episode_index),
                "seed": int(seed),
                "worker_index": int(worker_index),
                "worker_episode_index": int(local_episode_index),
                "return": float(ep_return),
                "mean_step_reward": float(mean_step_reward),
                "task_completion": float(task_completed_sum),
                "conflict_times": float(step_conflict_times),
                "invalid_action_rate": float(invalid_action_rate),
                "use_fp_rate": float(use_fp_rate),
                "steps": int(step_count),
                "sum_of_costs": float(episode_sum_of_costs),
                "termination_step": int(step_count),
            }
            results.append(result)
            if episode_done_cb is not None:
                episode_done_cb(dict(result))
            if progress_cb is not None:
                progress_cb(
                    f"[T{worker_index + 1}/{worker_count}] episode {episode_index + 1}/{total_episodes} | "
                    f"seed={seed} | task_completion={task_completed_sum:.1f} | "
                    f"conflict_times={step_conflict_times:.1f} | steps={step_count} | "
                    f"sum_of_costs={episode_sum_of_costs:.0f}"
                )
        return results
    finally:
        try:
            env.close()
        except Exception:
            pass


def _create_eval_agent_from_spec(
    eval_spec: dict[str, Any], env_cfg: EnvConfig
):
    mode = str(eval_spec.get("mode", "") or "").strip().lower()
    device = (
        str(eval_spec.get("device", "auto") or "auto").strip() or "auto"
    )
    obs_info = dict(eval_spec.get("obs_info", {}) or {})
    action_dim = int(eval_spec.get("action_dim", 0) or 0)
    n_agents = int(eval_spec.get("n_agents", 0) or 0)
    if mode == "model":
        model_path = str(eval_spec.get("model_path", "") or "").strip()
        run_config_path = (
            str(eval_spec.get("run_config_path", "") or "").strip()
        )
        spec = resolve_model_load_spec(model_path, run_config_path or None)
        agent = create_and_load_agent_from_model_spec(
            spec,
            obs_info,
            action_dim,
            n_agents,
            device=device,
            env_cfg=env_cfg,
            evaluation=True,
        )
    elif mode == "planner":
        agent = PlannerEvalAgent()
    else:
        raise ValueError(f"Unknown evaluation mode: {mode}")
    if hasattr(agent, "set_training_mode"):
        agent.set_training_mode(False)
    return agent


def _eval_group_process_main(
    eval_spec: dict[str, Any],
    env_cfg_dict: dict[str, Any],
    seed_group: list[tuple[int, int]],
    worker_index: int,
    worker_count: int,
    total_episodes: int,
    render_interval_s: float,
    result_queue: Any,
    cancel_event: Any,
    task_limit: int = 0,
) -> None:
    env_cfg = EnvConfig.from_dict(env_cfg_dict)

    def _agent_factory():
        return _create_eval_agent_from_spec(eval_spec, env_cfg)

    def _episode_done(result: dict[str, Any]) -> None:
        result_queue.put({"type": "episode", "result": dict(result)})

    try:
        _run_eval_seed_group(
            agent_factory=_agent_factory,
            env_cfg=env_cfg,
            seed_group=seed_group,
            worker_index=worker_index,
            worker_count=worker_count,
            total_episodes=total_episodes,
            render_interval_s=render_interval_s,
            stop_event=cancel_event,
            cancel_event=cancel_event,
            progress_cb=None,
            episode_done_cb=_episode_done,
            task_limit=task_limit,
        )
        result_queue.put(
            {"type": "worker_done", "worker_index": int(worker_index)}
        )
    except EvaluationStopped:
        result_queue.put(
            {
                "type": "worker_stopped",
                "worker_index": int(worker_index),
            }
        )
    except Exception:
        result_queue.put(
            {
                "type": "worker_error",
                "worker_index": int(worker_index),
                "traceback": traceback.format_exc(),
            }
        )


# ---------------------------------------------------------------------------
# High-level evaluation orchestrators
# ---------------------------------------------------------------------------
def evaluate_agent_rollout(
    agent_factory,
    env_cfg: EnvConfig,
    progress_cb,
    progress_state_cb=None,
    render_interval_s: float = 0.0,
    stop_event: Optional[threading.Event] = None,
    eval_threads: int = 1,
    eval_spec: Optional[dict[str, Any]] = None,
    task_limit: int = 0,
) -> dict[str, Any]:
    # ---- task_limit mode: relax episode step cap and expand seeds ----
    _original_max_episode_steps: int = int(
        env_cfg.eval.max_episode_steps or 300
    )
    _original_eval_episodes: int = int(env_cfg.eval_episodes or 5)
    # --------------------------------------------------------------
    seeds = build_eval_seed_list(env_cfg)
    seed_groups = split_eval_seeds_evenly(seeds, max(1, int(eval_threads)))
    worker_count = max(1, len(seed_groups))
    start_time = time.time()
    progress_lock = threading.Lock()
    cancel_event = threading.Event()
    worker_states: list[dict[str, Any]] = [
        {
            "worker_index": int(worker_index),
            "worker_label": f"T{worker_index + 1}",
            "status": "pending",
            "completed": 0,
            "total": len(group),
            "current_seed": None,
            "last_seed": None,
            "last_steps": 0,
            "completed_steps_total": 0,
            "last_task_completion": 0.0,
            "last_conflict_times": 0.0,
        }
        for worker_index, group in enumerate(seed_groups)
    ]
    episode_results: list[dict[str, Any]] = []

    def _emit_progress_state(phase: str) -> None:
        if progress_state_cb is None:
            return
        with progress_lock:
            workers_snapshot = [dict(item) for item in worker_states]
            completed_episodes = int(
                sum(
                    int(item.get("completed", 0)) for item in workers_snapshot
                )
            )
        total_episodes = len(seeds)
        percent = float(completed_episodes / float(max(1, total_episodes)))
        elapsed_sec = float(max(0.0, time.time() - start_time))
        completed_steps_total = int(
            sum(
                int(item.get("completed_steps_total", 0) or 0)
                for item in workers_snapshot
            )
        )
        active_workers = sum(
            1
            for item in workers_snapshot
            if str(item.get("status", "")) == "running"
        )
        rate = (
            float(completed_episodes / elapsed_sec)
            if elapsed_sec > 1e-6
            else 0.0
        )
        step_rate = (
            float(completed_steps_total / elapsed_sec)
            if elapsed_sec > 1e-6
            else 0.0
        )
        summary = (
            f"Progress {completed_episodes}/{total_episodes} "
            f"({percent * 100.0:.1f}%) | workers {active_workers}/{worker_count} | "
            f"elapsed {elapsed_sec:.1f}s | {rate:.2f} ep/s | {step_rate:.1f} step/s"
        )
        detail_parts: list[str] = []
        for item in workers_snapshot:
            label = str(item.get("worker_label", "T?"))
            status = str(item.get("status", "pending"))
            completed = int(item.get("completed", 0) or 0)
            total = int(item.get("total", 0) or 0)
            current_seed = item.get("current_seed")
            last_seed = item.get("last_seed")
            if status == "running" and current_seed is not None:
                detail_parts.append(
                    f"{label} {completed}/{total} seed={current_seed}"
                )
            elif status == "done" and last_seed is not None:
                detail_parts.append(
                    f"{label} done {completed}/{total} last_seed={last_seed}"
                )
            elif status == "stopped":
                detail_parts.append(f"{label} stopped")
            elif status == "error":
                detail_parts.append(f"{label} error")
            else:
                detail_parts.append(
                    f"{label} waiting {completed}/{total}"
                )
        progress_state_cb(
            {
                "phase": phase,
                "percent": percent,
                "completed_episodes": completed_episodes,
                "total_episodes": total_episodes,
                "eval_threads": worker_count,
                "active_workers": active_workers,
                "completed_steps_total": completed_steps_total,
                "steps_per_sec": step_rate,
                "elapsed_sec": elapsed_sec,
                "summary": summary,
                "detail": " | ".join(detail_parts),
                "seed_groups": summarize_seed_groups(seed_groups),
                "workers": workers_snapshot,
            }
        )

    def _mark_episode_done(result: dict[str, Any]) -> None:
        worker_index = int(result.get("worker_index", 0))
        with progress_lock:
            if 0 <= worker_index < len(worker_states):
                state = worker_states[worker_index]
                state["completed"] = int(state.get("completed", 0)) + 1
                state["last_seed"] = int(result.get("seed", 0))
                episode_steps = int(result.get("steps", 0))
                state["last_steps"] = episode_steps
                state["completed_steps_total"] = (
                    int(state.get("completed_steps_total", 0) or 0)
                    + episode_steps
                )
                state["last_task_completion"] = float(
                    result.get("task_completion", 0.0)
                )
                state["last_conflict_times"] = float(
                    result.get("conflict_times", 0.0)
                )
                group = seed_groups[worker_index]
                done_count = int(state["completed"])
                state["current_seed"] = (
                    int(group[done_count][1])
                    if done_count < len(group)
                    else None
                )
                state["status"] = (
                    "done" if done_count >= len(group) else "running"
                )
        _emit_progress_state("running")

    def _worker_entry(
        worker_index: int, group: list[tuple[int, int]]
    ) -> list[dict[str, Any]]:
        with progress_lock:
            state = worker_states[worker_index]
            state["status"] = "running"
            state["current_seed"] = int(group[0][1]) if group else None
        _emit_progress_state("running")
        try:
            return _run_eval_seed_group(
                agent_factory=agent_factory,
                env_cfg=env_cfg,
                seed_group=group,
                worker_index=worker_index,
                worker_count=worker_count,
                total_episodes=len(seeds),
                render_interval_s=render_interval_s,
                stop_event=stop_event,
                cancel_event=cancel_event,
                progress_cb=progress_cb,
                episode_done_cb=_mark_episode_done,
                task_limit=task_limit,
            )
        except EvaluationStopped:
            with progress_lock:
                worker_states[worker_index]["status"] = "stopped"
            cancel_event.set()
            _emit_progress_state("stopped")
            raise
        except Exception:
            with progress_lock:
                worker_states[worker_index]["status"] = "error"
            cancel_event.set()
            _emit_progress_state("error")
            raise

    _emit_progress_state("starting")
    try:
        if worker_count <= 1:
            episode_results.extend(_worker_entry(0, seed_groups[0]))
        else:
            if eval_spec is None:
                raise RuntimeError(
                    "Multi-worker evaluation is missing a serializable eval_spec."
                )
            mp_ctx = mp.get_context("spawn")
            result_queue = mp_ctx.Queue()
            cancel_mp_event = mp_ctx.Event()
            workers: list[Any] = []
            remaining_workers = worker_count
            env_cfg_dict = env_cfg_to_dict(env_cfg)
            try:
                for worker_index, group in enumerate(seed_groups):
                    with progress_lock:
                        state = worker_states[worker_index]
                        state["status"] = "running"
                        state["current_seed"] = (
                            int(group[0][1]) if group else None
                        )
                    _emit_progress_state("running")
                    worker = mp_ctx.Process(
                        target=_eval_group_process_main,
                        args=(
                            eval_spec,
                            env_cfg_dict,
                            group,
                            worker_index,
                            worker_count,
                            len(seeds),
                            render_interval_s,
                            result_queue,
                            cancel_mp_event,
                            task_limit,
                        ),
                        daemon=True,
                    )
                    worker.start()
                    workers.append(worker)
                while remaining_workers > 0:
                    if _should_stop_evaluation(stop_event):
                        cancel_mp_event.set()
                    try:
                        message = result_queue.get(timeout=0.1)
                    except queue.Empty:
                        if remaining_workers > 0 and all(
                            not worker.is_alive() for worker in workers
                        ):
                            break
                        continue
                    msg_type = (
                        str(message.get("type", "") or "").strip()
                    )
                    if msg_type == "episode":
                        result = dict(
                            message.get("result", {}) or {}
                        )
                        episode_results.append(result)
                        _mark_episode_done(result)
                    elif msg_type == "worker_done":
                        remaining_workers = max(0, remaining_workers - 1)
                    elif msg_type == "worker_stopped":
                        worker_index = int(
                            message.get("worker_index", 0)
                        )
                        with progress_lock:
                            if (
                                0
                                <= worker_index
                                < len(worker_states)
                            ):
                                worker_states[worker_index][
                                    "status"
                                ] = "stopped"
                        remaining_workers = max(
                            0, remaining_workers - 1
                        )
                        cancel_mp_event.set()
                        _emit_progress_state("stopped")
                    elif msg_type == "worker_error":
                        cancel_mp_event.set()
                        raise RuntimeError(
                            str(
                                message.get(
                                    "traceback", "Evaluation subprocess failed."
                                )
                            )
                        )
                if len(episode_results) < len(seeds):
                    if (
                        _should_stop_evaluation(stop_event)
                        or cancel_mp_event.is_set()
                    ):
                        raise EvaluationStopped("Evaluation terminated by the user.")
                    if task_limit <= 0:
                        raise RuntimeError(
                            f"Evaluation incomplete: expected {len(seeds)} episodes but "
                            f"only received {len(episode_results)} results."
                        )
            finally:
                cancel_mp_event.set()
                for worker in workers:
                    try:
                        worker.join(timeout=0.5)
                    except Exception:
                        pass
                for worker in workers:
                    try:
                        if worker.is_alive():
                            worker.terminate()
                    except Exception:
                        pass
                try:
                    result_queue.close()
                except Exception:
                    pass
    except Exception:
        cancel_event.set()
        raise
    finally:
        pass

    elapsed = time.time() - start_time
    episode_results = sorted(
        episode_results, key=lambda item: int(item.get("episode_index", 0))
    )
    returns = [
        float(item.get("return", 0.0)) for item in episode_results
    ]
    mean_step_rewards = [
        float(item.get("mean_step_reward", 0.0))
        for item in episode_results
    ]
    task_completion_totals = [
        float(item.get("task_completion", 0.0))
        for item in episode_results
    ]
    conflict_times_totals = [
        float(item.get("conflict_times", 0.0))
        for item in episode_results
    ]
    invalid_rates = [
        float(item.get("invalid_action_rate", 0.0))
        for item in episode_results
    ]
    use_fp_rates = [
        float(item.get("use_fp_rate", 0.0))
        for item in episode_results
    ]
    per_episode_seeds = [
        int(item.get("seed", 0)) for item in episode_results
    ]
    total_steps = int(
        sum(int(item.get("steps", 0)) for item in episode_results)
    )
    total_tasks_completed = float(sum(task_completion_totals))
    _emit_progress_state("done")

    env_eval_meta = {
        "num_agvs": env_cfg.eval.num_agvs,
        "map_size": env_cfg.eval.map_size,
        "max_episode_steps": _original_max_episode_steps,
        "seed": env_cfg.eval.seed,
        "render_mode": env_cfg.eval.render_mode,
        "render_interval_s": float(render_interval_s),
        "fov_size": env_cfg.fov_size,
        "kstep_conflict_check": env_cfg.kstep_conflict_check,
        "planner_type": env_cfg.planner.planner_type,
        "planner_overrides": dict(env_cfg.planner.overrides),
        "targets_on_shelf": bool(env_cfg.targets_on_shelf),
    }

    if task_limit > 0:
        # ===== task-limited mode: separate result structure =====
        termination_steps = [
            float(
                item.get("termination_step", item.get("steps", 0))
            )
            for item in episode_results
        ]
        tasks_completed_list = [
            float(item.get("task_completion", 0.0))
            for item in episode_results
        ]
        conflict_times_list = [
            float(item.get("conflict_times", 0.0))
            for item in episode_results
        ]

        return {
            "mode": "limited",
            "task_limit": int(task_limit),
            "eval_episodes": len(episode_results),
            "per_seed": [
                {
                    "seed": int(item.get("seed", 0)),
                    "termination_step": int(
                        item.get(
                            "termination_step", item.get("steps", 0)
                        )
                    ),
                    "tasks_completed": float(
                        item.get("task_completion", 0.0)
                    ),
                    "sum_of_costs": float(
                        item.get("sum_of_costs", 0.0)
                    ),
                    "conflict_times": float(
                        item.get("conflict_times", 0.0)
                    ),
                    "use_fp_rate": float(
                        item.get("use_fp_rate", 0.0)
                    ),
                }
                for item in episode_results
            ],
            "termination_step_mean": (
                mean(termination_steps) if termination_steps else 0.0
            ),
            "termination_step_std": (
                pstdev(termination_steps)
                if len(termination_steps) > 1
                else 0.0
            ),
            "tasks_completed_mean": (
                mean(tasks_completed_list)
                if tasks_completed_list
                else 0.0
            ),
            "conflict_times_mean": (
                mean(conflict_times_list)
                if conflict_times_list
                else 0.0
            ),
            "use_fp_mean": (
                mean(use_fp_rates) if use_fp_rates else 0.0
            ),
            "elapsed_sec": elapsed,
            "env_eval": env_eval_meta,
        }

    # ===== standard mode =====
    return {
        "mode": "standard",
        "episodes": len(episode_results),
        "return_mean": mean(returns) if returns else 0.0,
        "return_std": (
            pstdev(returns) if len(returns) > 1 else 0.0
        ),
        "step_reward_mean": (
            mean(mean_step_rewards) if mean_step_rewards else 0.0
        ),
        "task_completion_mean": (
            mean(task_completion_totals)
            if task_completion_totals
            else 0.0
        ),
        "task_completion_std": (
            pstdev(task_completion_totals)
            if len(task_completion_totals) > 1
            else 0.0
        ),
        "conflict_times_mean": (
            mean(conflict_times_totals)
            if conflict_times_totals
            else 0.0
        ),
        "conflict_times_std": (
            pstdev(conflict_times_totals)
            if len(conflict_times_totals) > 1
            else 0.0
        ),
        "invalid_action_rate_mean": (
            mean(invalid_rates) if invalid_rates else 0.0
        ),
        "use_fp_mean": (
            mean(use_fp_rates) if use_fp_rates else 0.0
        ),
        "per_episode_task_completion": [
            float(v) for v in task_completion_totals
        ],
        "per_episode_conflict_times": [
            float(v) for v in conflict_times_totals
        ],
        "per_episode_seeds": per_episode_seeds,
        "eval_threads": int(worker_count),
        "task_limit": 0,
        "seed_groups": [
            [int(seed) for _, seed in group] for group in seed_groups
        ],
        "elapsed_sec": elapsed,
        "env_eval": env_eval_meta,
    }


def evaluate_model(
    model_path: str,
    run_config_path: Optional[str],
    env_cfg: EnvConfig,
    device: str,
    progress_cb,
    progress_state_cb=None,
    render_interval_s: float = 0.0,
    stop_event: Optional[threading.Event] = None,
    eval_threads: int = 1,
    task_limit: int = 0,
) -> dict[str, Any]:
    spec = resolve_model_load_spec(
        model_path, run_config_path
    )
    algo_cfg = algo_config_from_dict(
        spec.algorithm, spec.algo_cfg_payload
    )
    resolved_device = str(device).strip() or "auto"
    algo_cfg.device = resolved_device

    try:
        if isinstance(
            algo_cfg, (SoftRHCRConfig, SoftRHCRMAPPOConfig)
        ):
            from SoftRHCR.algorithms.SoftRHCR.rhcr_utils import (
                apply_soft_rhcr_defaults,
            )

            apply_soft_rhcr_defaults(env_cfg, algo_cfg)
    except Exception:
        pass

    env = WarehouseEnv(**env_cfg.get_env_args(mode="eval"))
    try:
        obs_info = extract_obs_info(env)
        action_dim = env.action_spaces[env.possible_agents[0]].n
        n_agents = len(env.possible_agents)
    finally:
        try:
            env.close()
        except Exception:
            pass

    def _agent_factory() -> Any:
        agent = create_and_load_agent_from_model_spec(
            spec,
            obs_info,
            action_dim,
            n_agents,
            device=resolved_device,
            env_cfg=env_cfg,
            evaluation=True,
        )
        if hasattr(agent, "set_training_mode"):
            agent.set_training_mode(False)
        return agent

    eval_spec = {
        "mode": "model",
        "device": resolved_device,
        "algorithm": spec.algorithm,
        "algo_cfg_payload": spec.algo_cfg_payload,
        "obs_info": obs_info,
        "action_dim": int(action_dim),
        "n_agents": int(n_agents),
        "model_path": str(model_path),
        "run_config_path": (
            str(run_config_path)
            if run_config_path is not None
            else ""
        ),
    }

    rollout = evaluate_agent_rollout(
        _agent_factory,
        env_cfg,
        progress_cb,
        progress_state_cb=progress_state_cb,
        render_interval_s=render_interval_s,
        stop_event=stop_event,
        eval_threads=eval_threads,
        eval_spec=eval_spec,
        task_limit=task_limit,
    )
    return {
        "algorithm": spec.algorithm,
        "model_path": str(model_path),
        "run_name": Path(model_path).stem,
        **rollout,
    }


def evaluate_planner(
    env_cfg: EnvConfig,
    device: str,
    progress_cb,
    progress_state_cb=None,
    render_interval_s: float = 0.0,
    stop_event: Optional[threading.Event] = None,
    eval_threads: int = 1,
    task_limit: int = 0,
) -> dict[str, Any]:
    env = WarehouseEnv(**env_cfg.get_env_args(mode="eval"))
    try:
        obs_info = extract_obs_info(env)
        action_dim = env.action_spaces[env.possible_agents[0]].n
        n_agents = len(env.possible_agents)
    finally:
        try:
            env.close()
        except Exception:
            pass

    def _agent_factory() -> Any:
        agent = PlannerEvalAgent()
        if hasattr(agent, "set_training_mode"):
            agent.set_training_mode(False)
        return agent

    eval_spec = {
        "mode": "planner",
        "device": str(device).strip() or "auto",
        "obs_info": obs_info,
        "action_dim": int(action_dim),
        "n_agents": int(n_agents),
    }

    rollout = evaluate_agent_rollout(
        _agent_factory,
        env_cfg,
        progress_cb,
        progress_state_cb=progress_state_cb,
        render_interval_s=render_interval_s,
        stop_event=stop_event,
        eval_threads=eval_threads,
        eval_spec=eval_spec,
        task_limit=task_limit,
    )
    return {
        "algorithm": "follow_planner",
        "planner_type": env_cfg.planner.planner_type,
        "planner_overrides": dict(env_cfg.planner.overrides),
        **rollout,
    }


def env_cfg_to_dict(env_cfg: EnvConfig) -> dict[str, Any]:
    return asdict(env_cfg)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _print_evaluation_results(results: dict[str, Any]) -> None:
    import json
    print("\n" + "=" * 60)
    print("Evaluation results")
    print("=" * 60)
    print(json.dumps(results, ensure_ascii=False, indent=2))


def main(argv: Optional[list[str]] = None) -> int:
    run_cfg, meta = load_and_build_config("evaluate", argv=argv)
    model_path = str(meta.get("model_path", "") or "").strip()
    run_config_path = str(meta.get("run_config_path", "") or "").strip()
    extra = {
        "eval_mode": "model" if model_path != "" else "planner",
        "model_path": model_path,
        "run_config_path": run_config_path,
        "eval_episodes": int(meta.get("eval_episodes", 10)),
        "eval_threads": int(meta.get("eval_threads", 1)),
        "task_limit": int(meta.get("task_limit", 0)),
    }
    if not confirm_resolved_config("evaluate", run_cfg, extra=extra):
        return 1

    env_cfg = run_cfg.env
    device = str(meta.get("device", "auto")).strip() or "auto"
    render_interval_s = float(meta.get("render_interval_s", 0.0))
    eval_threads = max(1, int(meta.get("eval_threads", 1)))
    task_limit = max(0, int(meta.get("task_limit", 0)))

    def _progress_cb(message: str) -> None:
        print(message, flush=True)

    with _CliEvaluationProgress(
        total_episodes=int(env_cfg.eval_episodes),
    ) as progress:
        def _progress_state_cb(payload: dict[str, Any]) -> None:
            progress.update_state(payload)

        if model_path != "":
            results = evaluate_model(
                model_path=model_path,
                run_config_path=run_config_path if run_config_path else None,
                env_cfg=env_cfg,
                device=device,
                progress_cb=_progress_cb,
                progress_state_cb=_progress_state_cb,
                render_interval_s=render_interval_s,
                eval_threads=eval_threads,
                task_limit=task_limit,
            )
        else:
            results = evaluate_planner(
                env_cfg=env_cfg,
                device=device,
                progress_cb=_progress_cb,
                progress_state_cb=_progress_state_cb,
                render_interval_s=render_interval_s,
                eval_threads=eval_threads,
                task_limit=task_limit,
            )
        progress.finalize(results)

    _print_evaluation_results(results)

    # Save results to run_dir
    run_dir = Path(meta["run_dir"])
    results_dir = run_dir / "results"
    if results_dir.exists():
        import json
        results_path = results_dir / "evaluation_results.json"
        results_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nResults saved to: {results_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
