from __future__ import annotations

import sys
import time


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


class ProgressReporter:
    def __init__(self, label: str, total: int, stream=None, time_fn=None, width: int = 28):
        self.label = label
        self.total = max(1, total)
        self.stream = stream or sys.stderr
        self.time_fn = time_fn or time.time
        self.width = width
        self.started_at = self.time_fn()
        self.current = 0

    def update(self, current: int):
        self.current = max(0, min(current, self.total))
        elapsed = max(0.0, self.time_fn() - self.started_at)
        rate = self.current / elapsed if elapsed > 0 and self.current > 0 else 0.0
        remaining = (self.total - self.current) / rate if rate > 0 else 0.0
        pct = self.current / self.total
        filled = int(self.width * pct)
        bar = "#" * filled + "-" * (self.width - filled)
        line = (
            f"\r{self.label} [{bar}] {self.current}/{self.total} "
            f"{pct * 100:5.1f}% elapsed {format_duration(elapsed)} "
            f"ETA {format_duration(remaining)}"
        )
        self.stream.write(line)
        self.stream.flush()

    def finish(self):
        self.update(self.total)
        self.stream.write("\n")
        self.stream.flush()

    def step_callback(self, _pipeline, step: int, _timestep, _callback_kwargs):
        self.update(step + 1)
        return {}
