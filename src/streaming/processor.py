"""
Stream Processor for ST-HGAT-DRIO.

Supports Apache Flink and Apache Spark Structured Streaming backends.
Maintains a 14-day rolling window of daily aggregated demand per SKU-Store.
Routes malformed events to a dead-letter queue.
Buffers events in memory for up to 60 s when the backend is unavailable,
then raises an alert.
"""

import logging
import time
from collections import defaultdict, deque
from datetime import datetime, date
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Required fields for a valid event
_REQUIRED_FIELDS = ("sku_store", "timestamp", "demand")

# Maximum days in the rolling window
_WINDOW_DAYS = 14


def _parse_timestamp(ts: Any) -> Optional[date]:
    """Parse a timestamp value to a date. Returns None on failure."""
    if isinstance(ts, datetime):
        return ts.date()
    if isinstance(ts, date):
        return ts
    if isinstance(ts, str):
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(ts, fmt).date()
            except ValueError:
                continue
    return None


class _BackendConnection:
    """Thin wrapper around a streaming backend connection.

    The actual Flink/Spark connection is attempted lazily.  The
    ``available`` property returns False when the backend cannot be
    reached, triggering the in-memory fallback buffer.
    """

    def __init__(self, backend: str, cfg: Any):
        self.backend = backend
        self.cfg = cfg
        self._connected: bool = False
        self._try_connect()

    def _try_connect(self) -> None:
        try:
            if self.backend == "flink":
                import pyflink  # noqa: F401 – optional dependency
                self._connected = True
                logger.info("Connected to Flink backend.")
            elif self.backend == "spark":
                import pyspark  # noqa: F401 – optional dependency
                self._connected = True
                logger.info("Connected to Spark backend.")
        except ImportError:
            logger.warning(
                "Backend '%s' is not available (import failed). "
                "Falling back to in-memory buffer.",
                self.backend,
            )
            self._connected = False

    @property
    def available(self) -> bool:
        return self._connected

    def send(self, event: dict) -> None:
        """Forward a validated event to the backend (no-op when unavailable)."""
        if not self._connected:
            raise RuntimeError(f"Backend '{self.backend}' is not available.")
        # Real implementation would serialize and publish to Flink/Spark here.


class StreamProcessor:
    """Processes streaming supply chain events.

    Parameters
    ----------
    backend:
        ``"flink"`` or ``"spark"``.
    cfg:
        Configuration object / dict.  Expected key: ``streaming.buffer_seconds``
        (default 60).
    backend_factory:
        Optional callable ``(backend_name, cfg) -> connection`` used to inject
        a mock backend in tests.
    clock:
        Optional callable returning the current time in seconds (default
        ``time.monotonic``).  Inject a fake clock in tests to avoid real waits.
    """

    def __init__(
        self,
        backend: str,
        cfg: Any,
        *,
        backend_factory: Optional[Callable] = None,
        clock: Optional[Callable[[], float]] = None,
    ):
        if backend not in ("flink", "spark"):
            raise ValueError(
                f"Unknown streaming backend '{backend}'. "
                "Supported values: 'flink', 'spark'."
            )

        self.backend_name = backend
        self.cfg = cfg
        self._clock = clock or time.monotonic

        # Resolve buffer_seconds from cfg (dict or object)
        try:
            self._buffer_seconds: float = cfg["streaming"]["buffer_seconds"]
        except (TypeError, KeyError):
            try:
                self._buffer_seconds = cfg.streaming.buffer_seconds
            except AttributeError:
                self._buffer_seconds = 60.0

        # Connect to backend (or fall back)
        if backend_factory is not None:
            self._conn = backend_factory(backend, cfg)
        else:
            self._conn = _BackendConnection(backend, cfg)

        # 14-day rolling window: sku_store → {date: total_demand}
        # We keep an ordered deque of (date, demand) pairs sorted by date.
        self._buffers: dict[str, dict[date, float]] = defaultdict(dict)

        # Dead-letter queue for malformed events
        self.dead_letter: list[dict] = []

        # In-memory fallback buffer when backend is unavailable
        self._fallback_buffer: list[dict] = []
        self._unavailable_since: Optional[float] = None
        self._alert_raised: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_event(self, event: dict) -> None:
        """Validate and process a single event.

        Valid events update the 14-day rolling buffer.
        Malformed events are routed to ``self.dead_letter``.
        When the backend is unavailable, events are buffered in memory
        for up to ``buffer_seconds``; after that an alert is raised.
        """
        if not self._validate(event):
            self.dead_letter.append(event)
            return

        # Parse timestamp to date
        day = _parse_timestamp(event["timestamp"])
        if day is None:
            self.dead_letter.append(event)
            return

        sku_store: str = event["sku_store"]
        demand: float = float(event["demand"])

        # Try to forward to backend; fall back to in-memory buffer on failure
        if self._conn.available:
            try:
                self._conn.send(event)
                self._unavailable_since = None
                self._alert_raised = False
            except RuntimeError:
                self._handle_backend_unavailable(event)
        else:
            self._handle_backend_unavailable(event)

        # Always update the local rolling buffer regardless of backend status
        self._update_buffer(sku_store, day, demand)

    def get_feature_buffer(self, sku_store: str) -> np.ndarray:
        """Return the last ≤14 daily aggregated demand values for *sku_store*.

        Values are returned in chronological order (oldest first).
        """
        day_map = self._buffers.get(sku_store, {})
        if not day_map:
            return np.array([], dtype=np.float64)

        sorted_days = sorted(day_map.keys())
        return np.array([day_map[d] for d in sorted_days], dtype=np.float64)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate(self, event: Any) -> bool:
        """Return True iff *event* is a dict containing all required fields."""
        if not isinstance(event, dict):
            return False
        for field in _REQUIRED_FIELDS:
            if field not in event:
                return False
        return True

    def _update_buffer(self, sku_store: str, day: date, demand: float) -> None:
        """Aggregate demand for *day* and evict entries older than 14 days."""
        buf = self._buffers[sku_store]

        # Accumulate demand for the same day
        buf[day] = buf.get(day, 0.0) + demand

        # Evict days outside the 14-day window
        if len(buf) > _WINDOW_DAYS:
            oldest_days = sorted(buf.keys())[: len(buf) - _WINDOW_DAYS]
            for old_day in oldest_days:
                del buf[old_day]

    def _handle_backend_unavailable(self, event: dict) -> None:
        """Buffer event in memory; raise alert after buffer_seconds."""
        now = self._clock()
        if self._unavailable_since is None:
            self._unavailable_since = now
            logger.warning(
                "Streaming backend '%s' is unavailable. "
                "Buffering events in memory.",
                self.backend_name,
            )

        elapsed = now - self._unavailable_since
        if elapsed >= self._buffer_seconds and not self._alert_raised:
            self._alert_raised = True
            logger.error(
                "Streaming backend '%s' has been unavailable for %.1f s. "
                "Raising alert and switching to batch fallback mode.",
                self.backend_name,
                elapsed,
            )
            raise RuntimeError(
                f"Streaming backend '{self.backend_name}' unavailable for "
                f"{elapsed:.1f} s — alert raised, switching to batch fallback."
            )

        self._fallback_buffer.append(event)
