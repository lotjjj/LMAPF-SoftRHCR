"""Unified training logger wrapping TensorBoard SummaryWriter.

All training metrics (PPO loss, diagnostics, auxiliary loss, gradient
monitoring, and environment interaction) are managed by this single
logger and written to ``run_dir/run_name/log``.
"""

import os
from typing import Dict, Optional

from torch.utils.tensorboard import SummaryWriter


class TrainLogger:
    """Centralised TensorBoard logger for all training metrics.

    Parameters
    ----------
    log_dir : str
        Directory for TensorBoard event files.  Typically
        ``<run_dir>/<run_name>/log``.
    grad_monitor_interval : int
        How often (in update calls) to perform detailed gradient
        monitoring.  Defaults to 5.
    """

    def __init__(
        self,
        log_dir: str,
        grad_monitor_interval: int = 5,
    ) -> None:
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        self._update_count = 0
        self._grad_monitor_interval = int(grad_monitor_interval)

    # ------------------------------------------------------------------
    # Update-count / gradient-monitor bookkeeping
    # ------------------------------------------------------------------
    @property
    def update_count(self) -> int:
        return self._update_count

    def increment_update_count(self) -> None:
        self._update_count += 1

    def should_monitor_grads(self) -> bool:
        """Return *True* on every ``grad_monitor_interval``-th update."""
        return (
            self._update_count % self._grad_monitor_interval == 0
        )

    @property
    def grad_monitor_interval(self) -> int:
        return self._grad_monitor_interval

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def log_update_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log PPO update / optimiser / gradient metrics."""
        for key, value in metrics.items():
            if value is None:
                continue
            try:
                self.writer.add_scalar(key, float(value), step)
            except (TypeError, ValueError):
                pass

    def log_train_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log environment-interaction metrics (``train/`` prefix)."""
        for key, value in metrics.items():
            if value is None:
                continue
            try:
                self.writer.add_scalar(key, float(value), step)
            except (TypeError, ValueError):
                pass

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        try:
            self.writer.add_scalar(tag, float(value), step)
        except (TypeError, ValueError):
            pass

    def close(self) -> None:
        try:
            self.writer.flush()
            self.writer.close()
        except Exception:
            pass


def build_logger_from_run_dir(run_dir: str, run_name: str) -> TrainLogger:
    """Convenience factory: ``<run_dir>/<run_name>/log``."""
    log_dir = os.path.join(str(run_dir), "log")
    return TrainLogger(log_dir=log_dir)
