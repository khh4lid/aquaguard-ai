"""
Heartbeat / watchdog.

A safety device that fails silently is worse than no device, because people
stop watching once they trust it. This sends a periodic "system OK" ping so a
crashed/overheated/unplugged Pi is *noticed*. It pairs with:
  - systemd `Restart=on-failure` (in aquaguard.service) which restarts crashes,
  - this heartbeat, which tells you the system is alive *and* roughly how fast.

Call .tick() once per processed frame from the main loop; it self-throttles to
one ping every HEARTBEAT_EVERY seconds.
"""
import time
from datetime import datetime


class Heartbeat:
    def __init__(self, alert_manager, every_seconds, enabled=True):
        self.alerts = alert_manager
        self.every = max(every_seconds, 30)
        self.enabled = enabled
        self._last = time.time()
        self._frames = 0

    def tick(self, frames_delta=1):
        self._frames += frames_delta
        if not self.enabled:
            return
        now = time.time()
        if now - self._last >= self.every:
            window = now - self._last
            fps = self._frames / window if window > 0 else 0.0
            self._last = now
            self._frames = 0
            self.alerts.heartbeat(
                f"AquaGuard OK {datetime.now():%H:%M} — running ~{fps:.1f} FPS")
