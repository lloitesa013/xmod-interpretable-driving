"""X-MoD v2 longitudinal controller.

Re-parametrization core: the network predicts a *target speed* (a smooth scalar), and this
classical PID converts (target_speed, current_speed) -> mutually-exclusive (throttle, brake).
This is the field-standard control split (PDM-Lite / TransFuser-style) and is the variable the
X-MoD B2D negative result never tested: a raw-pedal head saturates into the safety/mobility
seesaw, whereas a target-speed head + this controller cannot output "full throttle into a stopped
car" or "full brake on an open road" unless the *target speed* itself is wrong.

Variant-independent: works with the original XMoDVLA and the 5090 separated-head variant alike.
Units: speeds in m/s. Outputs throttle in [0, throttle_max], brake in [0, brake_max].

Self-test: `python longitudinal_controller.py`
"""
from dataclasses import dataclass


@dataclass
class LongitudinalPID:
    kp: float = 1.0
    ki: float = 0.15
    kd: float = 0.05
    dt: float = 0.05               # CARLA fixed step at 20 Hz
    throttle_max: float = 0.75
    brake_max: float = 1.0
    integral_clamp: float = 5.0    # anti-windup
    stop_speed: float = 0.3        # below this target -> commit to a stop (m/s)
    deadband: float = 0.2          # |error| below this -> coast (m/s)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0

    def step(self, current_speed: float, target_speed: float) -> tuple:
        """Return (throttle, brake), mutually exclusive."""
        target_speed = max(0.0, float(target_speed))
        current_speed = max(0.0, float(current_speed))

        # Hard stop intent: if the policy asks for ~0 speed, brake to a halt and don't wind up.
        if target_speed <= self.stop_speed:
            self._integral = 0.0
            self._prev_error = -current_speed
            brake = self.brake_max if current_speed > self.stop_speed else 0.3
            return 0.0, min(self.brake_max, brake)

        error = target_speed - current_speed

        # Coast inside the deadband (avoids pedal chatter around the setpoint).
        if abs(error) <= self.deadband:
            self._prev_error = error
            return 0.0, 0.0

        self._integral = max(-self.integral_clamp,
                             min(self.integral_clamp, self._integral + error * self.dt))
        derivative = (error - self._prev_error) / self.dt
        self._prev_error = error

        u = self.kp * error + self.ki * self._integral + self.kd * derivative

        if u >= 0.0:
            return min(self.throttle_max, u), 0.0
        return 0.0, min(self.brake_max, -u)


def _selftest() -> None:
    # 1) Accelerate from rest toward 8 m/s: throttle should engage, brake stay 0.
    pid = LongitudinalPID()
    spd = 0.0
    for _ in range(80):
        thr, brk = pid.step(spd, 8.0)
        assert brk == 0.0
        spd += (thr * 4.0 - 0.1) * pid.dt    # crude plant: throttle -> accel
    assert spd > 6.0, f"failed to accelerate: {spd:.2f}"

    # 2) Front actor: target drops to 0 while moving -> brake, no throttle.
    pid.reset()
    thr, brk = pid.step(current_speed=8.0, target_speed=0.0)
    assert thr == 0.0 and brk > 0.5, (thr, brk)

    # 3) On setpoint -> coast (no chatter).
    pid.reset()
    thr, brk = pid.step(current_speed=8.0, target_speed=8.05)
    assert thr == 0.0 and brk == 0.0, (thr, brk)

    print("longitudinal_controller self-test OK")


if __name__ == "__main__":
    _selftest()
