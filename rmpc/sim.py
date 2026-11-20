"""Closed-loop episodes on a static occupancy map."""
import numpy as np

from .maps import Car, clearance, field_lookup, hazard_field
from .mpc import MPC
from .navigation import cost_to_go, guess_along, route

KAPPA = 100.0     # risk planner route cost: meters of detour per unit of hazard
BLOCKED = 20.0    # baseline route cost per meter inside obstacle cells (the constraint does the blocking)


def circle_radius(car, margin=0.2):
    return float(np.hypot(car.length / 6, car.width / 2)) + margin


def step(x, u, car, dt, sub=4):
    def f(s):
        return np.array([s[3] * np.cos(s[2]), s[3] * np.sin(s[2]),
                         s[3] / car.wheelbase * np.tan(u[1]), u[0]])
    h = dt / sub
    for _ in range(sub):
        k1 = f(x); k2 = f(x + h / 2 * k1); k3 = f(x + h / 2 * k2); k4 = f(x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x[3] = max(x[3], 0.0)
    return x


def front(x, car):
    return x[:2] + car.rear_to_front * np.array([np.cos(x[2]), np.sin(x[2])])


def center(x, car):
    return x[:2] + car.rear_to_center * np.array([np.cos(x[2]), np.sin(x[2])])


def run_episode(grid, start, goal, mode, param, car=Car(), t_max=20.0, goal_tol=2.0,
                corr_len=4.0, route_clearance=0.5, hazard_kw=None, **mpc_kw):
    """param is delta for mode='risk', tau for mode='threshold'.

    `spent` is the hazard accumulated along the executed path under the correlation-length
    model, in both modes; pred_risk = 1 - exp(-spent) is the map's own collision estimate.
    """
    haz = hazard_field(grid, car, corr_len, **(hazard_kw or {}))
    if mode == "risk":
        field = haz
        cell_cost = 1.0 + KAPPA * haz
    else:
        field = circle_radius(car) - clearance(grid, param)   # <= 0 keeps each circle clear
        cell_cost = np.where(field > 0, BLOCKED, np.where(field > -route_clearance, 3.0, 1.0))
    ctg, pred = cost_to_go(grid, cell_cost, goal)
    rtg = cost_to_go(grid, haz + 1e-5, goal)[0] if mode == "risk" else None
    mpc = MPC(grid.xs, grid.ys, mode, field, ctg, car=car, eps=0.0, risk_to_go=rtg, **mpc_kw)
    H = field_lookup(grid, haz)
    total = -np.log1p(-param) if mode == "risk" else 0.0
    spent = 0.0
    x = np.array(start, float)
    poses, times, fails, slacks, over = [x[:3].copy()], [], 0, [], 0
    t, still, reached, plan = 0.0, 0.0, False, None
    while t < t_max:
        if np.hypot(*(center(x, car) - goal)) < goal_tol:
            reached = True
            break
        guess = guess_along(route(grid, pred, center(x, car), goal), x, mpc.N, mpc.dt,
                            x[3], car.rear_to_center)
        out = mpc.solve(x, max(total - spent, 0.0), guess)
        times.append(out["solve_time"])
        slacks.append(out["slack"])
        if out["ok"]:
            plan, k = out["U"], 0
            u = plan[:, 0]
        else:
            fails += 1
            if plan is not None and k + 1 < plan.shape[1]:
                k += 1                          # keep following the last plan that met the constraint
                u = plan[:, k]
            else:
                over += 1                       # no plan meets the constraint: brake
                u = np.array([-6.0, 0.0])
        x_new = step(x.copy(), u, car, mpc.dt)
        f0, f1 = front(x, car), front(x_new, car)
        spent += float(H([(f0 + f1) / 2])[0]) * np.hypot(*(f1 - f0))
        x = x_new
        poses.append(x[:3].copy())
        t += mpc.dt
        still = still + mpc.dt if x[3] < 0.05 else 0.0
        if still >= 3.0:
            break
    return dict(poses=np.array(poses), reached=reached, time=t, fails=fails, over=over,
                max_slack=float(max(slacks, default=0.0)), solve_time=float(np.mean(times or [0])),
                spent=spent, budget=total, pred_risk=float(-np.expm1(-spent)))
