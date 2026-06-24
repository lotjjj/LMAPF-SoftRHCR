import numpy as np
import torch
from typing import Dict, Any, List
from SoftRHCR.modules.device import resolve_device


def _flatten_self_states(self_states) -> np.ndarray:
    def _flat(obj):
        if isinstance(obj, dict):
            out = []
            def _korder(k):
                key = str(k)
                if key == "path_info":
                    return (0, "")
                return (1, key)

            for k in sorted(obj.keys(), key=_korder):
                out.extend(_flat(obj[k]))
            return out
        if isinstance(obj, (list, tuple)):
            out = []
            for v in obj:
                out.extend(_flat(v))
            return out
        return [np.asarray(obj, dtype=np.float32).reshape(-1)]

    parts = _flat(self_states)
    if len(parts) == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

class ReplayBuffer:
    def __init__(self, buffer_size: int, n_agents: int, obs_shape: Dict, state_shape: int, action_dim: int, device: str = "auto"):
        self.buffer_size = buffer_size
        self.n_agents = n_agents
        self.device = resolve_device(device)
        self.ptr = 0
        self.count = 0
        
        self.obs = {
            'fov': np.zeros((buffer_size, n_agents, *obs_shape['fov']), dtype=np.float32),
            'msgs': np.zeros((buffer_size, n_agents, *obs_shape['msgs']), dtype=np.float32),
            'self_states': np.zeros((buffer_size, n_agents, obs_shape['self_states']), dtype=np.float32)
        }
        self.next_obs = {
            'fov': np.zeros((buffer_size, n_agents, *obs_shape['fov']), dtype=np.float32),
            'msgs': np.zeros((buffer_size, n_agents, *obs_shape['msgs']), dtype=np.float32),
            'self_states': np.zeros((buffer_size, n_agents, obs_shape['self_states']), dtype=np.float32)
        }
        self.actions = np.zeros((buffer_size, n_agents), dtype=np.int64)
        self.rewards = np.zeros((buffer_size, n_agents), dtype=np.float32)
        self.dones = np.zeros((buffer_size, n_agents), dtype=np.bool_)
        self.action_masks = np.ones((buffer_size, n_agents, action_dim), dtype=np.bool_)
        self.next_action_masks = np.ones((buffer_size, n_agents, action_dim), dtype=np.bool_)
        
        if state_shape > 0:
            self.states = np.zeros((buffer_size, state_shape), dtype=np.float32)
            self.next_states = np.zeros((buffer_size, state_shape), dtype=np.float32)
        else:
            self.states = None
            self.next_states = None

    def add(self, obs, actions, rewards, next_obs, dones, state=None, next_state=None, action_masks=None, next_action_masks=None):
        idx = self.ptr % self.buffer_size
        ss_dim = int(self.obs['self_states'].shape[-1])
        
        for agent_idx in range(self.n_agents):
            agent_id = f"agv_{agent_idx}"
            
            if agent_id in obs:
                self.obs['fov'][idx, agent_idx] = obs[agent_id]['fov']
                self.obs['msgs'][idx, agent_idx] = obs[agent_id]['msgs']
                ss_vec = _flatten_self_states(obs[agent_id]['self_states'])
                if int(ss_vec.size) != ss_dim:
                    raise RuntimeError(
                        f"self_states dim mismatch in ReplayBuffer.add: expected {ss_dim}, got {int(ss_vec.size)} for {agent_id}"
                    )
                self.obs['self_states'][idx, agent_idx] = ss_vec
                
                self.actions[idx, agent_idx] = actions.get(agent_id, 0)
                self.rewards[idx, agent_idx] = rewards.get(agent_id, 0.0)
                self.dones[idx, agent_idx] = dones.get(agent_id, True)
                
                if action_masks is not None and agent_id in action_masks:
                    self.action_masks[idx, agent_idx] = action_masks[agent_id]
                else:
                    self.action_masks[idx, agent_idx] = 1.0
            else:
                self.obs['fov'][idx, agent_idx] = 0
                self.obs['msgs'][idx, agent_idx] = 0
                self.obs['self_states'][idx, agent_idx] = 0
                self.actions[idx, agent_idx] = 0
                self.rewards[idx, agent_idx] = 0
                self.dones[idx, agent_idx] = True
                self.action_masks[idx, agent_idx] = 1.0

            if next_action_masks is not None and agent_id in next_action_masks:
                self.next_action_masks[idx, agent_idx] = next_action_masks[agent_id]
            else:
                self.next_action_masks[idx, agent_idx] = 1.0

            if agent_id in next_obs:
                self.next_obs['fov'][idx, agent_idx] = next_obs[agent_id]['fov']
                self.next_obs['msgs'][idx, agent_idx] = next_obs[agent_id]['msgs']
                next_ss_vec = _flatten_self_states(next_obs[agent_id]['self_states'])
                if int(next_ss_vec.size) != ss_dim:
                    raise RuntimeError(
                        f"next_self_states dim mismatch in ReplayBuffer.add: expected {ss_dim}, got {int(next_ss_vec.size)} for {agent_id}"
                    )
                self.next_obs['self_states'][idx, agent_idx] = next_ss_vec
            else:
                self.next_obs['fov'][idx, agent_idx] = 0
                self.next_obs['msgs'][idx, agent_idx] = 0
                self.next_obs['self_states'][idx, agent_idx] = 0
            
        if state is not None and self.states is not None:
            self.states[idx] = state
        if next_state is not None and self.next_states is not None:
            self.next_states[idx] = next_state
            
        self.ptr += 1
        self.count = min(self.count + 1, self.buffer_size)

    def sample(self, batch_size: int):
        indices = np.random.choice(self.count, batch_size, replace=False)
        
        batch = {
            'obs': {k: torch.tensor(v[indices], device=self.device) for k, v in self.obs.items()},
            'next_obs': {k: torch.tensor(v[indices], device=self.device) for k, v in self.next_obs.items()},
            'actions': torch.tensor(self.actions[indices], device=self.device),
            'rewards': torch.tensor(self.rewards[indices], device=self.device),
            'dones': torch.tensor(self.dones[indices], device=self.device),
            'action_masks': torch.tensor(self.action_masks[indices], device=self.device),
            'next_action_masks': torch.tensor(self.next_action_masks[indices], device=self.device),
        }
        if self.states is not None:
            batch['states'] = torch.tensor(self.states[indices], device=self.device)
        if self.next_states is not None:
            batch['next_states'] = torch.tensor(self.next_states[indices], device=self.device)
            
        return batch

class PPOBuffer:
    def __init__(self, n_steps: int, n_agents: int, obs_shape: Dict, action_dim: int, device: str = "auto"):
        self.n_steps = n_steps
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.device = resolve_device(device)
        self.ptr = 0
        
        self.obs = {
            'fov': np.zeros((n_steps, n_agents, *obs_shape['fov']), dtype=np.float32),
            'msgs': np.zeros((n_steps, n_agents, *obs_shape['msgs']), dtype=np.float32),
            'self_states': np.zeros((n_steps, n_agents, obs_shape['self_states']), dtype=np.float32)
        }
        self.next_obs = {
            'fov': np.zeros((n_steps, n_agents, *obs_shape['fov']), dtype=np.float32),
            'msgs': np.zeros((n_steps, n_agents, *obs_shape['msgs']), dtype=np.float32),
            'self_states': np.zeros((n_steps, n_agents, obs_shape['self_states']), dtype=np.float32)
        }
        self.actions = np.zeros((n_steps, n_agents), dtype=np.int64)
        self.logprobs = np.zeros((n_steps, n_agents), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_agents), dtype=np.float32)
        self.values = np.zeros((n_steps, n_agents), dtype=np.float32)
        self.next_values = np.zeros((n_steps, n_agents), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_agents), dtype=np.bool_)
        self.action_masks = np.ones((n_steps, n_agents, action_dim), dtype=np.bool_)
        self.next_action_masks = np.ones((n_steps, n_agents, action_dim), dtype=np.bool_)
        self.k_path_conflict = np.zeros((n_steps, n_agents), dtype=np.float32)
        self.task_completed = np.zeros((n_steps, n_agents), dtype=np.float32)
        self.conflicted = np.zeros((n_steps, n_agents), dtype=np.float32)
        self.use_fp = np.zeros((n_steps, n_agents), dtype=np.float32)

    def add(
        self,
        obs,
        actions,
        logprobs,
        rewards,
        values,
        dones,
        next_obs,
        action_masks=None,
        next_action_masks=None,
        k_path_conflict=None,
        task_completed=None,
        conflicted=None,
        use_fp=None,
    ):
        if self.ptr < self.n_steps:
            ss_dim = int(self.obs['self_states'].shape[-1])
            for i in range(self.n_agents):
                agent_id = f"agv_{i}"
                
                if agent_id in obs:
                    self.obs['fov'][self.ptr, i] = obs[agent_id]['fov']
                    self.obs['msgs'][self.ptr, i] = obs[agent_id]['msgs']
                    ss_vec = _flatten_self_states(obs[agent_id]['self_states'])
                    if int(ss_vec.size) != ss_dim:
                        raise RuntimeError(
                            f"self_states dim mismatch in PPOBuffer.add: expected {ss_dim}, got {int(ss_vec.size)} for {agent_id}"
                        )
                    self.obs['self_states'][self.ptr, i] = ss_vec
                    
                    self.actions[self.ptr, i] = actions.get(agent_id, 0)
                    self.logprobs[self.ptr, i] = logprobs.get(agent_id, 0.0)
                    self.rewards[self.ptr, i] = rewards.get(agent_id, 0.0)
                    if hasattr(values, "get"):
                        self.values[self.ptr, i] = values.get(agent_id, 0.0)
                    else:
                        self.values[self.ptr, i] = float(values) if values is not None else 0.0
                    self.dones[self.ptr, i] = dones.get(agent_id, True)
                    
                    if action_masks is not None and agent_id in action_masks:
                        self.action_masks[self.ptr, i] = action_masks[agent_id]
                    else:
                        self.action_masks[self.ptr, i] = 1.0
                    if k_path_conflict is not None and agent_id in k_path_conflict:
                        self.k_path_conflict[self.ptr, i] = float(k_path_conflict[agent_id])
                    else:
                        self.k_path_conflict[self.ptr, i] = 0.0
                    if task_completed is not None and agent_id in task_completed:
                        self.task_completed[self.ptr, i] = float(task_completed[agent_id])
                    else:
                        self.task_completed[self.ptr, i] = 0.0
                    if conflicted is not None and agent_id in conflicted:
                        self.conflicted[self.ptr, i] = float(conflicted[agent_id])
                    else:
                        self.conflicted[self.ptr, i] = 0.0
                    if use_fp is not None and agent_id in use_fp:
                        self.use_fp[self.ptr, i] = float(use_fp[agent_id])
                    else:
                        self.use_fp[self.ptr, i] = 0.0
                else:
                    self.obs['fov'][self.ptr, i] = 0
                    self.obs['msgs'][self.ptr, i] = 0
                    self.obs['self_states'][self.ptr, i] = 0
                    self.actions[self.ptr, i] = 0
                    self.logprobs[self.ptr, i] = 0
                    self.rewards[self.ptr, i] = 0
                    self.values[self.ptr, i] = 0
                    self.dones[self.ptr, i] = True
                    self.action_masks[self.ptr, i] = 1.0
                    self.k_path_conflict[self.ptr, i] = 0.0
                    self.task_completed[self.ptr, i] = 0.0
                    self.conflicted[self.ptr, i] = 0.0

                if next_action_masks is not None and agent_id in next_action_masks:
                    self.next_action_masks[self.ptr, i] = next_action_masks[agent_id]
                else:
                    self.next_action_masks[self.ptr, i] = 1.0

                if agent_id in next_obs:
                    self.next_obs['fov'][self.ptr, i] = next_obs[agent_id]['fov']
                    self.next_obs['msgs'][self.ptr, i] = next_obs[agent_id]['msgs']
                    next_ss_vec = _flatten_self_states(next_obs[agent_id]['self_states'])
                    if int(next_ss_vec.size) != ss_dim:
                        raise RuntimeError(
                            f"next_self_states dim mismatch in PPOBuffer.add: expected {ss_dim}, got {int(next_ss_vec.size)} for {agent_id}"
                        )
                    self.next_obs['self_states'][self.ptr, i] = next_ss_vec
                else:
                    self.next_obs['fov'][self.ptr, i] = 0
                    self.next_obs['msgs'][self.ptr, i] = 0
                    self.next_obs['self_states'][self.ptr, i] = 0
            self.ptr += 1

    def finalize_rollout(self):
        """Called after a rollout ends: fill ``next_values`` via a shift and drop the last step.

        ``next_values[t] = values[t+1]``; the last step has no available next
        value and is therefore discarded.
        """
        if self.ptr <= 1:
            # Not enough data to perform the shift
            self.ptr = 0
            return
        # shift: next_values[0:ptr-1] = values[1:ptr]
        self.next_values[0:self.ptr - 1] = self.values[1:self.ptr]
        # Drop the last step
        self.ptr -= 1

    def clear(self):
        self.ptr = 0

    def get_all(self):
        data = {
            'obs': {k: torch.tensor(v[:self.ptr], device=self.device) for k, v in self.obs.items()},
            'next_obs': {k: torch.tensor(v[:self.ptr], device=self.device) for k, v in self.next_obs.items()},
            'actions': torch.tensor(self.actions[:self.ptr], device=self.device),
            'logprobs': torch.tensor(self.logprobs[:self.ptr], device=self.device),
            'rewards': torch.tensor(self.rewards[:self.ptr], device=self.device),
            'values': torch.tensor(self.values[:self.ptr], device=self.device),
            'next_values': torch.tensor(self.next_values[:self.ptr], device=self.device),
            'dones': torch.tensor(self.dones[:self.ptr], device=self.device),
            'action_masks': torch.tensor(self.action_masks[:self.ptr], device=self.device),
            'next_action_masks': torch.tensor(self.next_action_masks[:self.ptr], device=self.device),
            'k_path_conflict': torch.tensor(self.k_path_conflict[:self.ptr], device=self.device),
            'task_completed': torch.tensor(self.task_completed[:self.ptr], device=self.device),
            'conflicted': torch.tensor(self.conflicted[:self.ptr], device=self.device),
            'use_fp': torch.tensor(self.use_fp[:self.ptr], device=self.device),
        }
        return data
