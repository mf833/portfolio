"""Single-axis reaction-wheel attitude control: PID gains by pole placement.

Plant:       I * theta'' = T_c + tau_d
Controller:  T_c = Kp*(b*theta_ref - theta) + Ki*int(e) - Kd*omega,  e = theta_ref - theta
             b = 1: PID on error (adds a closed-loop zero -> extra overshoot)
             b = 0: I-PD / setpoint weighting (no zero; response follows the placed poles)
Target poles: (s + a*wn)(s^2 + 2*zeta*wn*s + wn^2)
  Kd = I*wn*(a + 2*zeta),  Kp = I*wn^2*(1 + 2*a*zeta),  Ki = a*I*wn^3

Parameters below are illustrative (small-satellite scale), not from a specific mission.
Run: python3 reaction_wheel_sim.py
"""
import math

I = 0.10                    # kg*m^2
ZETA, WN, A = 0.7, 0.2, 4.0  # damping ratio, natural frequency [rad/s], third-pole factor
STEP_DEG = 10.0              # reference step
TAU_D = 1e-5                 # N*m constant disturbance torque (SRP/drag order of magnitude)

Kd = I * WN * (A + 2 * ZETA)
Kp = I * WN**2 * (1 + 2 * A * ZETA)
Ki = A * I * WN**3


def simulate(theta_ref, tau_d, b=1.0, t_end=80.0, dt=0.01):
    """RK4 on x = [theta, omega, integral of error]. Returns lists t, theta, torque."""
    def f(x):
        th, om, ie = x
        e = theta_ref - th
        tc = Kp * (b * theta_ref - th) + Ki * ie - Kd * om
        return [om, (tc + tau_d) / I, e], tc

    x, t, out = [0.0, 0.0, 0.0], 0.0, []
    while t <= t_end + 1e-9:
        _, tc = f(x)
        out.append((t, x[0], tc))
        k1, _ = f(x)
        k2, _ = f([x[i] + dt / 2 * k1[i] for i in range(3)])
        k3, _ = f([x[i] + dt / 2 * k2[i] for i in range(3)])
        k4, _ = f([x[i] + dt * k3[i] for i in range(3)])
        x = [x[i] + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(3)]
        t += dt
    return out


def metrics(out, ref):
    peak = max(th for _, th, _ in out)
    settle = max(t for t, th, _ in out if abs(th - ref) > 0.02 * ref)
    return (peak - ref) / ref * 100, settle, max(abs(tc) for *_, tc in out)


if __name__ == "__main__":
    ref = math.radians(STEP_DEG)
    print(f"Kp={Kp:.4g} N*m/rad  Ki={Ki:.4g} N*m/(rad*s)  Kd={Kd:.4g} N*m*s/rad")
    for b, name in ((1.0, "PID on error"), (0.0, "I-PD (setpoint weighting)")):
        os_, ts, tmax = metrics(simulate(ref, 0.0, b), ref)
        print(f"{name:27s} step {STEP_DEG:g} deg: overshoot {os_:5.1f}%, 2% settling {ts:5.1f} s, "
              f"peak commanded torque {tmax * 1e3:.2f} mN*m")
    dist = simulate(0.0, TAU_D)
    worst = max(abs(th) for _, th, _ in dist)
    print(f"Disturbance {TAU_D:g} N*m: worst error {math.degrees(worst) * 3600:.1f} arcsec, "
          f"error at t=80 s {math.degrees(dist[-1][1]) * 3600:.3f} arcsec (integral action)")
