"""Causal AE50 brightness allocation between CLID exposure and onboard light."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


Meter = Callable[[float, float], float]  # commanded light, exposure ms -> Bayer gray mean DN


@dataclass(frozen=True)
class AutoLightPolicy:
    light_min: float = 0.1
    preferred_ms: float = 8.0
    rate_hz: float = 3.0
    max_step: float = 0.1

    def __post_init__(self) -> None:
        if not math.isfinite(self.light_min) or not 0 <= self.light_min < 1:
            raise ValueError('light_min must be finite in [0, 1)')
        if not math.isfinite(self.preferred_ms) or not 1 <= self.preferred_ms <= 16:
            raise ValueError('preferred_ms must be finite in [1, 16]')
        if not math.isfinite(self.rate_hz) or self.rate_hz <= 0:
            raise ValueError('rate_hz must be finite and positive')
        if not math.isfinite(self.max_step) or not 0 < self.max_step <= 1:
            raise ValueError('max_step must be finite in (0, 1]')


def exposure_for_mean(meter: Meter, light: float, target: float,
                      low: float, high: float) -> float:
    """Bounded inversion; exposure scaling and camera response are monotone."""
    if target <= meter(light, low):
        return low
    if target >= meter(light, high):
        return high
    for _ in range(16):
        middle = (low + high) / 2
        if meter(light, middle) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


class AutoLightController:
    def __init__(self, policy: AutoLightPolicy) -> None:
        self.policy = policy
        self.light = policy.light_min
        self.exposure_ms = 4.0
        self.last_light_change_ns: int | None = None

    def step(self, requested_ms: float, next_stamp_ns: int, meter: Meter) -> dict[str, float | str]:
        """Choose controls for the next frame using only the current frame's model."""
        policy = self.policy
        requested_ms = min(16.0, max(1.0, requested_ms))
        target = meter(self.light, requested_ms)
        preferred = policy.preferred_ms
        if target <= meter(policy.light_min, preferred):
            ideal_light = policy.light_min
        else:
            # Full CLID response need not be strictly monotone in commanded
            # light because of C1/C2; scan rather than binary-search it.
            candidates = [policy.light_min]
            candidates.extend(min(1.0, i / 100) for i in range(
                math.floor(policy.light_min * 100) + 1, 101))
            ideal_light = next((light for light in candidates
                                if meter(light, preferred) >= target), 1.0)
        changed = abs(ideal_light - self.light) > 1e-12
        interval_ns = math.ceil(1e9 / policy.rate_hz)
        rate_held = changed and self.last_light_change_ns is not None and (
            next_stamp_ns - self.last_light_change_ns < interval_ns)
        next_light = self.light
        step_limited = False
        if changed and not rate_held:
            delta = ideal_light - self.light
            step_limited = abs(delta) > policy.max_step + 1e-12
            next_light = min(1.0, max(policy.light_min, self.light +
                             math.copysign(min(abs(delta), policy.max_step), delta)))
            if abs(next_light - self.light) > 1e-12:
                self.last_light_change_ns = next_stamp_ns
        if target <= meter(next_light, preferred):
            next_ms = exposure_for_mean(meter, next_light, target, 1.0, preferred)
            exposure_limit = 'none'
        elif next_light < 1.0 - 1e-12:
            # Respect the user's 8-ms preference while light can still ramp.
            next_ms = preferred
            exposure_limit = 'preferred'
        else:
            next_ms = exposure_for_mean(meter, next_light, target, preferred, 16.0)
            exposure_limit = 'max' if next_ms >= 16.0 - 1e-9 else 'none'
        predicted = meter(next_light, next_ms)
        self.light, self.exposure_ms = next_light, next_ms
        return {'target_dn': target, 'ideal_light': ideal_light, 'next_light': next_light,
                'next_ms': next_ms, 'predicted_dn': predicted,
                'light_limit': 'rate' if rate_held else 'step' if step_limited else 'none',
                'exposure_limit': exposure_limit}
