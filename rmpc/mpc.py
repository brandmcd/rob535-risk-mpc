"""Kinematic-bicycle MPC on an occupancy map, solved with CasADi + IPOPT.

mode="risk":      sum_k G(front_k) * v_k * dt + R(front_N) <= budget, where G is
                  maps.hazard_field, R is the least hazard from the horizon's end to the goal,
                  and budget is what is left of -log(1 - delta) after the distance driven.
mode="threshold": cells with p > tau are obstacles; `field` is circle radius minus the
                  distance to the nearest obstacle, and must stay <= eps at the centers of
                  three circles covering the car.

Both modes track a cost-to-go field (navigation.cost_to_go) computed on their own map.
The two fields are baked into B-spline lookup tables, so a new MPC is built per map.
"""
import time

import casadi as ca
import numpy as np

from .maps import Car

FEAS_TOL = 1e-4


class MPC:
    def __init__(self, xs, ys, mode, field, ctg, car=Car(), N=20, dt=0.2, v_max=10.0,
                 eps=0.02, method="bspline", risk_to_go=None):
        self.mode, self.car, self.N, self.dt, self.eps = mode, car, N, dt, eps
        axes = [np.asarray(xs, float), np.asarray(ys, float)]
        F = ca.interpolant("field", method, axes, np.asarray(field, float).ravel(order="F"))
        C = ca.interpolant("ctg", method, axes, np.asarray(ctg, float).ravel(order="F"))
        R = (ca.interpolant("rtg", method, axes, np.asarray(risk_to_go, float).ravel(order="F"))
             if risk_to_go is not None else (lambda pt: 0))
        opti = ca.Opti()
        X = opti.variable(4, N + 1)   # rear axle x, y, heading, speed
        U = opti.variable(2, N)       # accel, steer
        x0 = opti.parameter(4)
        budget = opti.parameter()
        slack = opti.variable()

        def f(x, u):
            return ca.vertcat(x[3] * ca.cos(x[2]), x[3] * ca.sin(x[2]),
                              x[3] / car.wheelbase * ca.tan(u[1]), u[0])

        def along(x, d):
            return x[:2] + d * ca.vertcat(ca.cos(x[2]), ca.sin(x[2]))

        cost, risk, cons = 0, 0, []
        opti.subject_to(X[:, 0] == x0)
        for k in range(N):
            xk, uk = X[:, k], U[:, k]
            k1 = f(xk, uk); k2 = f(xk + dt / 2 * k1, uk)
            k3 = f(xk + dt / 2 * k2, uk); k4 = f(xk + dt * k3, uk)
            opti.subject_to(X[:, k + 1] == xk + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
            nxt = X[:, k + 1]
            cost += C(along(nxt, car.rear_to_center))
            cost += 0.05 * uk[0] ** 2 + 2.0 * uk[1] ** 2
            if k > 0:
                cost += 0.5 * ca.sumsqr(U[:, k] - U[:, k - 1])
            if mode == "risk":
                # trapezoid over start, middle and end of the step so a narrow hazard band
                # between knots (2 m apart at 10 m/s) is not skipped
                f0, f1 = along(xk, car.rear_to_front), along(nxt, car.rear_to_front)
                ds = 0.5 * (xk[3] + nxt[3]) * dt
                risk += (0.25 * F(f0) + 0.5 * F(0.5 * (f0 + f1)) + 0.25 * F(f1)) * ds
            else:
                for off in (-car.length / 3, 0.0, car.length / 3):
                    cons.append(F(along(nxt, car.rear_to_center + off)))
        cost += 5.0 * C(along(X[:, N], car.rear_to_center))
        if mode == "risk":
            risk += R(along(X[:, N], car.rear_to_front))
            opti.subject_to(risk <= budget + slack)
            viol = risk - budget
        else:
            cons = ca.vertcat(*cons)
            opti.subject_to(cons <= eps + slack)
            viol = ca.mmax(cons) - eps
        opti.subject_to(slack >= 0)
        cost += 1e3 * slack
        opti.subject_to(opti.bounded(0, X[3, :], v_max))
        opti.subject_to(opti.bounded(-6.0, U[0, :], 2.5))
        opti.subject_to(opti.bounded(-0.5, U[1, :], 0.5))
        opti.minimize(cost)
        opti.solver("ipopt", {"print_time": False},
                    {"print_level": 0, "max_iter": 150, "tol": 1e-4, "sb": "yes",
                     "acceptable_tol": 1e-2, "acceptable_constr_viol_tol": 1e-4})
        self.opti, self.X, self.U, self.x0, self.budget = opti, X, U, x0, budget
        self.slack, self.risk_expr, self.viol = slack, risk, viol
        self._guess = None

    def reset(self):
        self._guess = None

    def solve(self, x0, budget=0.0, X_guess=None):
        """Returns the plan; ok is True only for a converged solve whose plan meets the
        constraint (slack and violation within FEAS_TOL). X_guess is used when there is
        no previous solution to shift."""
        o = self.opti
        o.set_value(self.x0, x0)
        o.set_value(self.budget, budget)
        if self._guess is None:
            Xg = X_guess if X_guess is not None else np.repeat(np.asarray(x0, float)[:, None], self.N + 1, axis=1)
            o.set_initial(self.X, Xg)
            o.set_initial(self.U, 0)
        else:
            Xg, Ug = self._guess
            o.set_initial(self.X, np.hstack([Xg[:, 1:], Xg[:, -1:]]))
            o.set_initial(self.U, np.hstack([Ug[:, 1:], Ug[:, -1:]]))
        t = time.perf_counter()
        try:
            o.solve()
        except RuntimeError:
            pass
        d = o.debug
        Xs, Us = d.value(self.X), d.value(self.U)
        slack, viol = float(d.value(self.slack)), float(d.value(self.viol))
        status = d.stats()["return_status"]
        ok = (status in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
              and slack < FEAS_TOL and viol < FEAS_TOL and np.all(np.isfinite(Us)))
        self._guess = (Xs, Us) if ok else None
        return dict(ok=ok, X=Xs, U=Us, slack=slack, viol=viol, status=status,
                    iters=d.stats()["iter_count"], solve_time=time.perf_counter() - t)
