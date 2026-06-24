from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np

from LMAPFEnv import WarehouseEnv
from LMAPFEnv.algorithms.path_planners import PlannerPolicy
from SoftRHCR.config.configBase import RunConfig
from SoftRHCR.config.registry import normalize_algorithm_token
from SoftRHCR.config.config_loader import load_and_build_config
from SoftRHCR.modules.model_io import (
    build_algo_config_from_model_spec,
    create_and_load_agent_from_model_spec,
    resolve_model_load_spec,
)
from SoftRHCR.scripts.cli_review import confirm_resolved_config


def build_cfg_from_cli(argv: Optional[list[str]] = None) -> dict[str, Any]:
    """Legacy-compatible entry: read the runtime config from the CLI and return it as a dict."""
    run_cfg, meta = load_and_build_config("experiment", argv=argv)
    return {"run_cfg": run_cfg, "meta": meta}


@dataclass
class Experiment:
    run_cfg: Any
    agent: Any
    trainer: Any
    run_name: str
    obs_info: dict[str, Any]
    observation_snapshot: dict[str, Any]
    overlays: dict[str, Any]


@dataclass
class RuntimeTrainerContext:
    env_cfg: Any
    algo_cfg: Any
    agent: Any
    observation_snapshot: dict[str, Any]

    def close(self) -> None:
        return None


class PlannerRuntimeAgent:
    def __init__(self) -> None:
        self._policy: Optional[PlannerPolicy] = None

    def bind_env(self, env: WarehouseEnv) -> None:
        planner = getattr(env, "path_planner", None)
        if planner is None:
            raise RuntimeError("The environment has no path_planner enabled; cannot run the planner runtime experiment.")
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


def _normalize_msgs_mode(mode: Optional[str]) -> Optional[str]:
    normalized = str(mode or "").strip().lower()
    if normalized in ("single", "dual"):
        return normalized
    return None


def _extract_runtime_obs_info(env: WarehouseEnv) -> dict[str, Any]:
    obs_info: dict[str, Any] = {}
    agent_id = env.possible_agents[0]
    obs_space = env.observation_spaces[agent_id]
    obs_info["fov"] = obs_space.spaces["fov"].shape
    if hasattr(obs_space, "spaces") and "msgs" in obs_space.spaces:
        obs_info["msgs"] = obs_space.spaces["msgs"].shape
    else:
        obs_info["msgs"] = (
            int(getattr(env, "kstep_conflict_check", 0)),
            obs_info["fov"][1],
            obs_info["fov"][2],
        )
    self_state_dim = 0
    if hasattr(obs_space, "spaces") and "self_states" in obs_space.spaces:
        for sub in obs_space.spaces["self_states"].spaces.values():
            if hasattr(sub, "shape"):
                self_state_dim += int(np.prod(sub.shape))
    obs_info["self_states"] = int(self_state_dim)
    obs_info["map_width"] = int(getattr(env, "width", 1))
    obs_info["map_height"] = int(getattr(env, "height", 1))
    return obs_info


def build_experiment(run_cfg: RunConfig, meta: dict[str, Any]) -> Experiment:
    run_env_cfg = run_cfg.env
    root_dir = Path(meta["run_dir"])
    model_path = str(meta.get("model_path", "") or "").strip()
    normalized_algorithm = normalize_algorithm_token(run_cfg.algorithm) or str(
        run_cfg.algorithm
    ).strip().lower()

    env = WarehouseEnv(**run_env_cfg.get_env_args(mode="runtime"))
    n_agents = len(env.possible_agents)
    action_dim = int(env.action_spaces["agv_0"].n)
    obs_info = _extract_runtime_obs_info(env)

    if model_path == "":
        if normalized_algorithm != "follow_planner":
            env.close()
            raise ValueError(
                "ExpConfig.model_path must not be empty; to run a planner runtime "
                "experiment set algorithm to follow_planner and leave model_path empty."
            )
        runtime_run_cfg = RunConfig(
            algorithm="follow_planner",
            env=run_env_cfg,
            algo=run_cfg.algo,
            observation={},
        )
        agent = PlannerRuntimeAgent()
        agent.set_training_mode(False)
        observation_snapshot: dict[str, Any] = {
            "adapter": "",
            "msgs_mode": "planner",
        }
        env.close()
    else:
        spec = resolve_model_load_spec(
            model_path,
            str(meta.get("run_config_path", "") or "") if meta.get("run_config_path") else None,
        )
        explicit_msgs_mode = _normalize_msgs_mode(meta.get("msgs_mode"))
        if explicit_msgs_mode is not None:
            spec = replace(
                spec, msgs_mode_hint=explicit_msgs_mode, msgs_mode_explicit=True
            )

        runtime_algo_cfg = build_algo_config_from_model_spec(
            spec,
            device=run_cfg.algo.device,
            env_cfg=run_env_cfg,
            override_msgs_mode=explicit_msgs_mode,
        )
        runtime_run_cfg = RunConfig(
            algorithm=str(spec.algorithm),
            env=run_env_cfg,
            algo=runtime_algo_cfg,
            observation={},
        )
        agent = create_and_load_agent_from_model_spec(
            spec,
            obs_info,
            action_dim,
            n_agents,
            device=runtime_algo_cfg.device,
            env_cfg=run_env_cfg,
            evaluation=True,
        )
        agent.set_training_mode(False)
        env.close()

        observation_snapshot = {
            "adapter": (
                type(agent.obs_adapter).__name__
                if hasattr(agent, "obs_adapter")
                else ""
            ),
            "msgs_mode": str(
                getattr(agent, "_loaded_soft_rhcr_msgs_mode", None)
                or getattr(runtime_algo_cfg, "soft_rhcr_msgs_mode", "single")
                or "single"
            ),
        }

    trainer = RuntimeTrainerContext(
        env_cfg=run_env_cfg,
        algo_cfg=runtime_run_cfg.algo,
        agent=agent,
        observation_snapshot=observation_snapshot,
    )

    return Experiment(
        run_cfg=runtime_run_cfg,
        agent=agent,
        trainer=trainer,
        run_name=str(meta.get("run_name", "") or "runtime"),
        obs_info=obs_info,
        observation_snapshot=observation_snapshot,
        overlays={
            "msgs_mode": (
                _normalize_msgs_mode(meta.get("msgs_mode"))
                if model_path != ""
                else None
            ),
            "runtime_mode": "planner" if model_path == "" else "model",
        },
    )


def _build_activations(
    run_config_path: str,
    model_path: Optional[str],
    cfg: dict[str, Any],
) -> Experiment:
    overrides = dict(cfg or {})
    if run_config_path:
        overrides["run_config_path"] = run_config_path
    if model_path is not None:
        overrides["model_path"] = model_path
    # Build the argv list passed to load_and_build_config
    argv: list[str] = []
    for key, value in overrides.items():
        if value is None:
            continue
        argv.extend([f"--{key.replace('_', '-')}", str(value)])
    run_cfg, meta = load_and_build_config("experiment", argv=argv or None)
    return build_experiment(run_cfg, meta)


def main(argv: Optional[list[str]] = None) -> int:
    run_cfg, meta = load_and_build_config("experiment", argv=argv)
    model_path = str(meta.get("model_path", "") or "").strip()
    extra = {
        "runtime_mode": "model" if model_path != "" else "planner",
        "model_path": model_path,
        "run_config_path": str(meta.get("run_config_path", "") or "").strip(),
        "msgs_mode": str(meta.get("msgs_mode", "") or "").strip(),
    }
    if not confirm_resolved_config("experiment", run_cfg, extra=extra):
        return 1
    build_experiment(run_cfg, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
