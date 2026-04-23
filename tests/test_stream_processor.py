"""
Tests for StreamProcessor (src/streaming/processor.py).

Covers:
- Property 13: Streaming Rolling Window Bound (Hypothesis, 100 examples)
- Property 14: Malformed Event Dead-Letter Routing (Hypothesis, 100 examples)
- Unit test 11.3: Backend unavailability buffering and alert after 60 s
"""

import logging
from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.streaming.processor import StreamProcessor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_DATE = date(2024, 1, 1)


def _make_cfg(buffer_seconds: float = 60.0) -> dict:
    return {"streaming": {"backend": "flink", "buffer_seconds": buffer_seconds}}


def _unavailable_backend_factory(backend: str, cfg):
    """Returns a mock connection that is always unavailable."""
    conn = MagicMock()
    conn.available = False
    conn.send.side_effect = RuntimeError("Backend unavailable")
    return conn


def _available_backend_factory(backend: str, cfg):
    """Returns a mock connection that is always available."""
    conn = MagicMock()
    conn.available = True
    conn.send.return_value = None
    return conn


def _make_event(sku_store: str, day: date, demand: float) -> dict:
    return {
        "sku_store": sku_store,
        "timestamp": day.isoformat(),
        "demand": demand,
    }


# ---------------------------------------------------------------------------
# Property 13: Streaming Rolling Window Bound
# Feature: st-hgat-drio, Property 13: streaming rolling window bound
# Validates: Requirements 8.4
# ---------------------------------------------------------------------------

@given(
    num_days=st.integers(min_value=15, max_value=60),
    demand_values=st.lists(
        st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=15,
        max_size=60,
    ),
)
@settings(max_examples=100)
def test_rolling_window_bound(num_days, demand_values):
    # Feature: st-hgat-drio, Property 13: streaming rolling window bound
    # Validates: Requirements 8.4
    #
    # Send one event per day for num_days days (always > 14).
    # The buffer must never exceed 14 entries.
    cfg = _make_cfg()
    processor = StreamProcessor(
        "flink", cfg, backend_factory=_available_backend_factory
    )

    sku = "SKU_A-STORE_1"
    n = min(num_days, len(demand_values))

    for i in range(n):
        day = _BASE_DATE + timedelta(days=i)
        processor.process_event(_make_event(sku, day, demand_values[i]))

    buf = processor.get_feature_buffer(sku)
    assert len(buf) <= 14, (
        f"Buffer length {len(buf)} exceeds 14-day window after {n} days of events."
    )


# ---------------------------------------------------------------------------
# Property 14: Malformed Event Dead-Letter Routing
# Feature: st-hgat-drio, Property 14: malformed event dead-letter routing
# Validates: Requirements 8.3
# ---------------------------------------------------------------------------

# Strategy: generate events that are missing at least one required field.
_REQUIRED_FIELDS = ["sku_store", "timestamp", "demand"]


def _malformed_event_strategy():
    """Generate a dict that is missing at least one required field."""
    full_event = st.fixed_dictionaries({
        "sku_store": st.text(min_size=1, max_size=20),
        "timestamp": st.dates(
            min_value=date(2020, 1, 1), max_value=date(2025, 12, 31)
        ).map(lambda d: d.isoformat()),
        "demand": st.floats(min_value=0.0, max_value=1000.0, allow_nan=False),
    })

    # Pick a non-empty subset of fields to drop
    @st.composite
    def _drop_fields(draw):
        event = draw(full_event)
        fields_to_drop = draw(
            st.lists(
                st.sampled_from(_REQUIRED_FIELDS),
                min_size=1,
                max_size=len(_REQUIRED_FIELDS),
                unique=True,
            )
        )
        for f in fields_to_drop:
            event.pop(f, None)
        return event

    return _drop_fields()


@given(event=_malformed_event_strategy())
@settings(max_examples=100)
def test_malformed_event_dead_letter_routing(event):
    # Feature: st-hgat-drio, Property 14: malformed event dead-letter routing
    # Validates: Requirements 8.3
    #
    # Any event missing a required field must end up in dead_letter and
    # must NOT appear in any feature buffer.
    cfg = _make_cfg()
    processor = StreamProcessor(
        "flink", cfg, backend_factory=_available_backend_factory
    )

    processor.process_event(event)

    # Must be in dead-letter queue
    assert len(processor.dead_letter) == 1, (
        f"Expected malformed event in dead-letter queue, got {processor.dead_letter}"
    )

    # Must not appear in any feature buffer
    for sku_store in processor._buffers:
        buf = processor.get_feature_buffer(sku_store)
        assert len(buf) == 0, (
            f"Malformed event unexpectedly updated buffer for '{sku_store}'"
        )


# ---------------------------------------------------------------------------
# Unit test 11.3: Backend unavailability buffering and alert after 60 s
# Validates: Requirements 8.5
# ---------------------------------------------------------------------------

def test_backend_unavailable_buffers_events():
    """Events are buffered in memory when the backend is unavailable."""
    cfg = _make_cfg(buffer_seconds=60.0)
    fake_time = [0.0]

    def fake_clock():
        return fake_time[0]

    processor = StreamProcessor(
        "flink",
        cfg,
        backend_factory=_unavailable_backend_factory,
        clock=fake_clock,
    )

    # Send a few events before the 60-second threshold
    for i in range(3):
        day = _BASE_DATE + timedelta(days=i)
        processor.process_event(_make_event("SKU_B-STORE_2", day, float(i + 1)))

    # Events should be in the fallback buffer, not dead-letter
    assert len(processor._fallback_buffer) == 3, (
        f"Expected 3 buffered events, got {len(processor._fallback_buffer)}"
    )
    assert len(processor.dead_letter) == 0


def test_backend_unavailable_raises_alert_after_timeout(caplog):
    """After 60 s of unavailability, an alert is raised (RuntimeError)."""
    cfg = _make_cfg(buffer_seconds=60.0)
    fake_time = [0.0]

    def fake_clock():
        return fake_time[0]

    processor = StreamProcessor(
        "flink",
        cfg,
        backend_factory=_unavailable_backend_factory,
        clock=fake_clock,
    )

    # First event — starts the unavailability timer
    processor.process_event(_make_event("SKU_C-STORE_3", _BASE_DATE, 10.0))

    # Advance fake clock past the 60-second threshold
    fake_time[0] = 61.0

    # Next event should trigger the alert
    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="unavailable"):
            processor.process_event(
                _make_event("SKU_C-STORE_3", _BASE_DATE + timedelta(days=1), 20.0)
            )

    # An error-level log entry must have been emitted
    assert any(
        record.levelno >= logging.ERROR for record in caplog.records
    ), "Expected an error-level alert log entry."


def test_backend_unavailable_alert_raised_only_once():
    """The alert RuntimeError is raised only once after the timeout is crossed."""
    cfg = _make_cfg(buffer_seconds=60.0)
    fake_time = [0.0]

    def fake_clock():
        return fake_time[0]

    processor = StreamProcessor(
        "flink",
        cfg,
        backend_factory=_unavailable_backend_factory,
        clock=fake_clock,
    )

    # First event — starts the unavailability timer at t=0
    processor.process_event(_make_event("SKU_D-STORE_4", _BASE_DATE, 5.0))

    # Advance past the 60-second threshold
    fake_time[0] = 61.0

    # Second event — triggers the alert (first time past threshold)
    with pytest.raises(RuntimeError):
        processor.process_event(
            _make_event("SKU_D-STORE_4", _BASE_DATE + timedelta(days=1), 5.0)
        )

    # Subsequent events should NOT raise again (alert already raised)
    processor.process_event(_make_event("SKU_D-STORE_4", _BASE_DATE + timedelta(days=2), 5.0))
    processor.process_event(_make_event("SKU_D-STORE_4", _BASE_DATE + timedelta(days=3), 5.0))
