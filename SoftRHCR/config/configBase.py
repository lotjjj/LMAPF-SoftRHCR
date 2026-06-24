from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar, Union

@dataclass
class EnvParams:
    """Concrete per-mode runtime parameters for the environment."""
    num_agvs: int = 5
    max_episode_steps: int = 200
    map_size: str = "long"
    seed: int = 42
    render_mode: Optional[str] = None
    render_interval_s: float = 0.0

DEFAULT_PLANNER_TYPE: Optional[str] = "AStar"

@dataclass
class PlannerConfig:
    planner_type: Optional[str] = DEFAULT_PLANNER_TYPE
    overrides: Dict[str, Any] = field(default_factory=lambda: dict())

    def resolved_planner_args(self) -> Optional[Dict[str, Any]]:
        if self.planner_type is None:
            return None

        overrides = dict(self.overrides or {})
        if "ecbs_w" in overrides and "w" not in overrides:
            overrides["w"] = overrides.pop("ecbs_w")

        cfg = None
        try:
            from LMAPFEnv.configBase import PLANNER_REGISTRY
            spec = dict(PLANNER_REGISTRY or {}).get(self.planner_type)
            if spec is not None:
                cfg = spec.default_config()
        except Exception:
            cfg = None
        if cfg is None:
            try:
                from LMAPFEnv.configBase import get_default_planner_config
                cfg = get_default_planner_config(self.planner_type)
            except Exception:
                return overrides if overrides else None
        if cfg is None:
            return None
        cfg = cfg.with_overrides(overrides)
        kwargs = dict(cfg.to_planner_kwargs() or {})
        # Preserve passthrough overrides that are not modeled by the simulator's
        # typed planner config, such as observation-planner split arguments.
        for key, value in overrides.items():
            if key not in kwargs:
                kwargs[key] = value
        return kwargs if kwargs else None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PlannerConfig":
        data = dict(data or {})
        planner_type = data.get("planner_type", data.get("path_planner", DEFAULT_PLANNER_TYPE))
        if isinstance(planner_type, str) and planner_type.lower() == "none":
            planner_type = None
        overrides = data.get("overrides", data.get("planner_overrides", data.get("planner_args", {})))
        if overrides is None:
            overrides = {}
        return cls(planner_type=planner_type, overrides=dict(overrides))

@dataclass
class EnvConfig:
    """Top-level environment config: shared observation/planner params plus three
    independent runtime instances.

    Shared fields (fov_size / kstep_conflict_check / targets_on_shelf / planner)
    are identical for every caller, while the train / eval / runtime modes each
    keep their own ``EnvParams`` to preserve per-scenario differences such as
    num_agvs / map_size / seed / horizon / render settings.
    """
    fov_size: int = 11
    kstep_conflict_check: int = 10
    targets_on_shelf: bool = True
    planner: PlannerConfig = field(default_factory=PlannerConfig)

    train: EnvParams = field(default_factory=
    lambda: EnvParams(seed=114514, render_mode=None)
    )
    eval: EnvParams = field(default_factory=
    lambda: EnvParams(seed=42, render_mode=None, max_episode_steps=300)
    )
    runtime: EnvParams = field(default_factory=
    lambda: EnvParams(seed=43, render_mode=None, max_episode_steps=1000)
    )
    eval_episodes: int = 10

    @property
    def path_planner(self) -> Optional[str]:
        return self.planner.planner_type

    @path_planner.setter
    def path_planner(self, v: Optional[str]) -> None:
        if isinstance(v, str) and v.lower() == "none":
            v = None
        self.planner.planner_type = v

    def get_env_args(self, mode: str = "train") -> Dict:
        """Return the env constructor args for ``mode``, merging the shared
        fields with the corresponding ``EnvParams``."""
        normalized = str(mode or "train").strip().lower()
        if normalized not in ("train", "eval", "runtime"):
            raise ValueError(f"Unsupported env mode: {mode!r} (expected: train / eval / runtime)")
        params = getattr(self, normalized)

        return {
            "num_agvs": params.num_agvs,
            "fov_size": self.fov_size,
            "render_mode": params.render_mode,
            "map_size": params.map_size,
            "max_episode_steps": params.max_episode_steps,
            "kstep_conflict_check": self.kstep_conflict_check,
            "path_planner": self.planner.planner_type,
            "targets_on_shelf": bool(self.targets_on_shelf),
            "planner_args": self.planner.resolved_planner_args(),
        }

    def get_params(self, mode: str) -> EnvParams:
        normalized = str(mode or "train").strip().lower()
        if normalized not in ("train", "eval", "runtime"):
            raise ValueError(f"Unsupported env mode: {mode!r} (expected: train / eval / runtime)")
        return getattr(self, normalized)

    def get_eval_seeds(self) -> List[int]:
        return [self.eval.seed + i for i in range(self.eval_episodes)]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "EnvConfig":
        data = dict(data or {})

        cfg = cls()
        if "fov_size" in data:
            cfg.fov_size = int(data["fov_size"])
        if "kstep_conflict_check" in data:
            cfg.kstep_conflict_check = int(data["kstep_conflict_check"])
        if "targets_on_shelf" in data:
            cfg.targets_on_shelf = bool(data["targets_on_shelf"])

        cfg.planner = PlannerConfig.from_dict(data.get("planner"))
        if "path_planner" in data and data.get("planner") is None:
            cfg.path_planner = data.get("path_planner")

        for mode_key in ("train", "eval", "runtime"):
            mode_d = data.get(mode_key)
            if isinstance(mode_d, dict):
                setattr(
                    cfg,
                    mode_key,
                    EnvParams(**{k: v for k, v in mode_d.items() if k in EnvParams.__dataclass_fields__}),
                )

        if "eval_episodes" in data:
            cfg.eval_episodes = int(data["eval_episodes"])

        legacy_w_train = data.get("train", {}).get("ecbs_w") if isinstance(data.get("train"), dict) else None
        legacy_w_eval = data.get("eval", {}).get("ecbs_w") if isinstance(data.get("eval"), dict) else None
        legacy_w_runtime = data.get("runtime", {}).get("ecbs_w") if isinstance(data.get("runtime"), dict) else None
        legacy_w = next(
            (value for value in (legacy_w_train, legacy_w_eval, legacy_w_runtime) if value is not None),
            None,
        )
        if legacy_w is not None and isinstance(cfg.planner.planner_type, str) and "ecbs" in cfg.planner.planner_type.lower():
            if "w" not in cfg.planner.overrides:
                cfg.planner.overrides["w"] = float(legacy_w)

        return cfg

@dataclass
class CommonConfig:
    """Algorithm and training parameters (excludes environment parameters)."""
    seed: int = 42
    device: str = "auto"
    norm_type: str = "bn"
    extractor_backbone: str = "default"
    norm_after_concat: str = "ln"
    profile: bool = True
    profile_interval_s: float = 5.0
    horizon_len: int = 400
    max_train_steps: int = 1_000_000
    
    # --- Reinforcement-learning hyperparameters (shared across algorithms) ---
    gamma: float = 0.99
    lr: float = 3e-4
    lr_schedule: Optional[str] = "linear"  # linear | cosine | None
    min_lr: float = 2e-5
    buffer_size: int = 20000
    # 0 => use the entire rollout as a single batch; >0 => update in chunks of batch_size.
    batch_size: int = 0
    target_update_interval: int = 100

    save_interval_steps: int = 100000
    model_dir: Optional[str] = None

    # --- Optimizer ---
    optimizer: Literal["Adam", "AdamW"] = "AdamW"
    weight_decay: float = 1e-4

    def resolved_horizon_len(self) -> int:
        horizon_len = int(self.horizon_len)
        if horizon_len <= 0:
            raise ValueError("horizon_len must be > 0")
        return horizon_len

    def resolved_target_total_steps(self) -> int:
        horizon_len = self.resolved_horizon_len()
        max_train_steps = int(self.max_train_steps)
        if max_train_steps <= 0:
            raise ValueError("max_train_steps must be > 0")
        horizon_k = max(1, int((max_train_steps + (horizon_len // 2)) // horizon_len))
        return int(horizon_k * horizon_len)

    def resolved_save_interval_steps(self) -> int:
        return int(self.save_interval_steps)

@dataclass
class MAPPOConfig(CommonConfig):
    """Algorithm-specific parameters for MAPPO."""
    ppo_epoch: int = 4
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    huber_delta: float = 10.0
    use_clipped_value_loss: bool = True
    use_gae: bool = True
    gae_lambda: float = 0.95
    critic_mode: Literal["per_agent", "team"] = "per_agent"

@dataclass
class IPPOConfig(CommonConfig):
    """Algorithm-specific parameters for IPPO."""
    ppo_epoch: int = 4
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    use_gae: bool = True
    gae_lambda: float = 0.95
    critic_mode: Literal["homogeneous"] = "homogeneous"

RewardModeType = Literal["legacy", "aggressive"]

@dataclass
class GateBlendConfig(IPPOConfig):
    """GateBlend: gate-based blending between planner following and PPO actor"""
    reward_mode: RewardModeType = "legacy"
    lambda1: float = 0.1
    lambda2: float = 0.1
    lambda3: float = 0.02
    gate_no_ppo_grad: bool = False
    specialized_lambda_anneal: bool = True
    specialized_lambda_min: float = 0.0
    bce_lambda_anneal: bool = True
    bce_lambda_min: float = 0.0
    bce_lambda_exp_k: float = 10.0
    specialized_huber_delta: float = 1.0
    gate_hidden_dim: int = 64
    hard_tau: float = 0.8
    fp_soft_beta: float = 2.0
    kpc_horizon: int = 10
    kpc_exp_beta: float = 10.0
    conflict_sparse_gate_supervision: bool = False

@dataclass
class GateBlendMAPPOConfig(MAPPOConfig):
    """GateBlend-MAPPO: GateBlend with MAPPO critic"""
    reward_mode: RewardModeType = "legacy"
    lambda1: float = 0.1 # gate_kl
    lambda2: float = 0.1 # gate_bce
    lambda3: float = 0.02 # gate_ent
    gate_no_ppo_grad: bool = False
    specialized_lambda_anneal: bool = True
    specialized_lambda_min: float = 0.0
    bce_lambda_anneal: bool = True
    bce_lambda_min: float = 0.0
    bce_lambda_exp_k: float = 10.0
    specialized_huber_delta: float = 1.0
    gate_hidden_dim: int = 128
    hard_tau: float = 0.8
    fp_soft_beta: float = 2.0
    kpc_horizon: int = 10
    kpc_exp_beta: float = 10.0
    conflict_sparse_gate_supervision: bool = True

@dataclass
class FollowPlannerConfig(CommonConfig):
    pass


PlannerAuxLossType = Literal["none", "consistency", "kl"]
PolicyUpdateModeType = Literal["all", "on_policy", "kpc_nonzero"]
SoftRHCRMsgsModeType = Literal["single", "dual"]


@dataclass
class SoftRHCRConfig(IPPOConfig):
    """SoftRHCR: Rule-based gate with IPPO collision avoidance"""
    reward_mode: RewardModeType = "aggressive"
    soft_rhcr_L: int = 2
    soft_rhcr_k: int = 10
    force_rl_prob_start: float = 0.8
    force_rl_prob_end: float = 0.2
    decay_steps_ratio: float = 0.5
    soft_rhcr_msgs_mode: SoftRHCRMsgsModeType = "dual"
    # Which samples enter the PPO update:
    #   all         -> all samples;
    #   on_policy   -> only use_fp=0 (RL) samples;
    #   kpc_nonzero -> only samples with k_path_conflict != 0.
    policy_update_mode: PolicyUpdateModeType = "on_policy"
    # Planner auxiliary loss mode: lower-bound consistency constraint or legacy KL imitation.
    planner_aux_loss: PlannerAuxLossType = "consistency"
    kl_coef: float = 0.0  # weight of the legacy planner KL/BC auxiliary term
    kl_coef_end: Optional[float] = None  # annealing end-point for the KL/BC weight (None keeps the start value)
    fp_consistency_coef: float = 0.1  # weight of the planner consistency constraint
    fp_consistency_coef_end: Optional[float] = None  # annealing end-point; None keeps the start value
    fp_consistency_pmin: float = 0.6  # minimum required probability for the planner action on use_fp steps
    fp_consistency_safe_only: bool = True  # apply the constraint only on safe planner steps where k_path_conflict == 0

@dataclass
class SoftRHCRMAPPOConfig(MAPPOConfig):
    """SoftRHCR-MAPPO: Rule-based gate with MAPPO collision avoidance"""
    reward_mode: RewardModeType = "aggressive"
    soft_rhcr_L: int = 2
    soft_rhcr_k: int = 10
    force_rl_prob_start: float = 0.8
    force_rl_prob_end: float = 0.0
    decay_steps_ratio: float = 0.5
    soft_rhcr_msgs_mode: SoftRHCRMsgsModeType = "single"
    # Which samples enter the PPO update:
    #   all         -> all samples;
    #   on_policy   -> only use_fp=0 (RL) samples;
    #   kpc_nonzero -> only samples with k_path_conflict != 0.
    policy_update_mode: PolicyUpdateModeType = "on_policy"
    # Planner auxiliary loss mode: lower-bound consistency constraint or legacy KL imitation.
    planner_aux_loss: PlannerAuxLossType = "consistency"
    kl_coef: float = 0.1  # weight of the legacy planner KL/BC auxiliary term
    kl_coef_end: Optional[float] = 0.001  # annealing end-point for the KL/BC weight
    fp_consistency_coef: float = 0.1  # weight of the planner consistency constraint
    fp_consistency_coef_end: Optional[float] = 0.01  # annealing end-point; None keeps the start value
    fp_consistency_pmin: float = 0.6  # minimum required probability for the planner action on use_fp steps
    fp_consistency_safe_only: bool = True  # apply the constraint only on safe planner steps where k_path_conflict == 0


AlgoConfigT = TypeVar("AlgoConfigT", bound=CommonConfig)


def _filter_dataclass_kwargs(cls: Type[Any], data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    d = dict(data or {})
    fields = getattr(cls, "__dataclass_fields__", {})
    return {k: v for k, v in d.items() if k in fields}


def normalize_algorithm_name(algorithm: str) -> str:
    algo = str(algorithm).strip().lower()
    return algo


def algo_config_from_dict(algorithm: str, data: Optional[Dict[str, Any]]) -> CommonConfig:
    """Delegate to registry.py to avoid duplicating algorithm-to-config mapping."""
    from SoftRHCR.config.registry import algo_config_from_dict as _registry_algo_config_from_dict
    return _registry_algo_config_from_dict(algorithm, data)


@dataclass
class RunConfig:
    algorithm: str
    env: EnvConfig = field(default_factory=EnvConfig)
    algo: CommonConfig = field(default_factory=CommonConfig)
    observation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        def _to(obj: Any) -> Any:
            if is_dataclass(obj):
                return asdict(obj)
            if isinstance(obj, dict):
                return {k: _to(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to(v) for v in obj]
            return obj

        return {
            "algorithm": str(self.algorithm),
            "env": _to(self.env),
            "algo": _to(self.algo),
            "observation": _to(self.observation),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunConfig":
        d = dict(data or {})
        algorithm = normalize_algorithm_name(str(d.get("algorithm", d.get("algo_name", d.get("algo", "")))).strip() or "mappo")

        env_d = d.get("env", d.get("env_cfg", {}))
        algo_d = d.get("algo", d.get("algo_cfg", {}))
        observation_d = d.get("observation", {})
        env_cfg = EnvConfig.from_dict(env_d if isinstance(env_d, dict) else {})
        algo_cfg = algo_config_from_dict(algorithm, algo_d if isinstance(algo_d, dict) else {})
        return cls(
            algorithm=algorithm,
            env=env_cfg,
            algo=algo_cfg,
            observation=dict(observation_d) if isinstance(observation_d, dict) else {},
        )
