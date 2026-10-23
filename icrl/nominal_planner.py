"""Fixed-grid trajectory planner with the same interface as ``DS_Policy``.

See docs/nominal_planner.md for the maths. In short: on the environment's own
time grid (step ``dt``), find the shortest-duration waypoint sequence from start
to goal that is smooth and feasible under the learned constraint network, then
replay it as actions. ``dscl.py`` only needs ``predict`` and ``modulate_with_NN``.
"""
import math
import zlib
from dataclasses import dataclass

import numpy as np
import torch as th
from numpy import dtype
from scipy import sparse

from icrl.optimizer import solve_trajectory_least_squares


def solve_penalty(initial, objective, constraints, *, max_nfev=60, weight=100.0, passes=3):
    """Least squares on [objective; weight * constraint violations], raising the weight until feasible.

    ``objective(z)`` returns (residuals, sparse Jacobian). Each constraint is a dict with
    ``evaluate(z) -> (values, sparse Jacobian)`` where values >= 0 means satisfied, a
    ``scale`` that makes violations comparable between constraints, and a ``tolerance`` in the
    constraint's own units. Raises ValueError if no pass satisfies every tolerance.
    """
    z = np.asarray(initial, dtype=float)
    for attempt in range(passes):
        w, cache = weight * 10 ** attempt, {}

        def evaluate(values):
            if "x" not in cache or not np.array_equal(values, cache["x"]):
                residual, jacobian = objective(values)
                residuals, jacobians = [residual], [sparse.csr_matrix(jacobian)]
                for c in constraints:
                    value, derivative = c["evaluate"](values)
                    active = sparse.diags((np.asarray(value) < 0) * w / c["scale"])
                    residuals.append(w / c["scale"] * np.minimum(value, 0.0))
                    jacobians.append(active @ sparse.csr_matrix(derivative))
                cache.update(x=values.copy(), residual=np.concatenate(residuals),
                             jacobian=sparse.vstack(jacobians, format="csr"))
                if not np.isfinite(cache["residual"]).all():
                    raise ValueError("non-finite residual")
            return cache

        result = solve_trajectory_least_squares(
            lambda v: evaluate(v)["residual"], z, jac=lambda v: evaluate(v)["jacobian"],
            tr_solver="lsmr", x_scale="jac", max_nfev=max_nfev, ftol=1e-5, xtol=1e-5, gtol=1e-5)
        z = result.x
        if all(np.min(c["evaluate"](z)[0]) >= -c["tolerance"] for c in constraints):
            return z
    raise ValueError("no feasible solution found")


def net_cost_and_obs_grad(net, obs):
    """Return the network cost and its gradient with respect to the full observation.

    The network sees only the entries / features chosen with ``-cosd``; autograd differentiates
    through that selection, so unused observation entries get gradient zero.
    cost = 1 - net(x); cost > threshold is infeasible.
    """
    if net.current_obs_mean is not None:
        raise ValueError("planner needs un-normalized observations (use -dno without -cn)")
    obs = np.atleast_2d(np.asarray(obs, dtype=np.float64))
    obs_t = th.tensor(obs, dtype=th.float32, device=net.device, requires_grad=True)
    acs_t = th.zeros(len(obs), net.acs_dim, device=net.device)
    # autograd runs the chain rule through select_appropriate_dims (speed / distance features included)
    cost = 1 - net(net.select_appropriate_dims(th.cat([obs_t, acs_t], dim=-1)))
    grad = th.autograd.grad(cost.sum(), obs_t)[0].cpu().numpy()
    return cost.detach().cpu().numpy().reshape(-1).astype(np.float64), grad.astype(np.float64)


@dataclass
class Plan:
    positions: np.ndarray   # (N, 3) waypoints on the environment's time grid
    actions: np.ndarray     # (N-1, 3) = displacement / step_size, inside [-1, 1]
    feasible: bool
    min_steps: int = 0      # steps of the fastest feasible plan; len(actions) is larger when slack was added


class Trajectory_Optimizer:
    """Plans minimum-duration, smooth, constraint-feasible paths on a fixed time grid."""

    def __init__(self, env=None, *, margin=0.02, w_len=1.0, w_acc=5.0, v_max=None, a_max=None,
                 box_fraction=0.97, bump=0.08, growth=(1.1, 1.25, 1.5, 2.0, 3.0), bisect_iters=3, duration_slack=(1.0, 2.5),
                 max_evaluations=60, constraint_weight=100.0, verbose=False):
        if env is None:
            from gym_panda.envs.KukaCup1 import KukaCup1
            env = KukaCup1()
        self.step_size, self.dt = float(env.step_size), float(env.dt)
        self.workspace_low = np.asarray(env.workspace_low, dtype=float)
        self.workspace_high = np.asarray(env.workspace_high, dtype=float)
        self.max_steps = int(env.max_steps)
        self.margin, self.w_len, self.w_acc = margin, w_len, w_acc
        self.v_max, self.a_max = v_max, a_max
        self.box_fraction, self.bump = box_fraction, bump
        self.growth, self.bisect_iters, self.duration_slack = growth, bisect_iters, duration_slack
        self.max_evaluations, self.constraint_weight = max_evaluations, constraint_weight
        self.verbose = verbose
        self._net, self._version = None, 0
        self._plans, self._cursor = {}, {}
        self.failures = 0

    # ------------------------------------------------------------------ DS_Policy interface
    def modulate_with_NN(self, gamma_function, rho=None, yita=None):
        """Store the learned constraint. ``gamma_function`` is ``ConstraintNet.build_gamma_with_grad_for_ds``."""
        net = getattr(gamma_function, "__self__", None)
        if net is None or not hasattr(net, "prepare_data"):
            raise ValueError("pass the bound method ConstraintNet.build_gamma_with_grad_for_ds")
        self._net, self._version = net, self._version + 1
        self._plans.clear()
        self._cursor.clear()

    def modulate_with_NN_with_refer_point(self, gamma_function, refer_point=None, rho=None, yita=None):
        """Same as ``modulate_with_NN``; the DS reference point has no meaning for a planner."""
        self.modulate_with_NN(gamma_function, rho)

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, observation, state=None, deterministic=True, **_):
        """Replay the plan for each environment row; plan (or re-plan) when the state leaves it."""
        observation = np.atleast_2d(np.asarray(observation, dtype=np.float64))
        actions = np.zeros((len(observation), 3))
        for row, obs in enumerate(observation):
            position, goal, displacement = obs[:3], obs[:3] + obs[3:6], obs[6:9]
            cursor = self._cursor.get(row)
            if cursor is None or not cursor.accepts(position, goal):
                cursor = _Cursor(self._plan_cached(position, goal, displacement), goal)
                self._cursor[row] = cursor
            actions[row] = cursor.next_action(position, self.step_size)
        return actions, state

    # ------------------------------------------------------------------ planning
    def _plan_cached(self, start, goal, v0):
        key = tuple(np.round(np.r_[start, goal, v0], 6))
        if key not in self._plans:
            # seeded by the problem and the constraint version: reproducible, new draw every iteration
            rng = np.random.RandomState(zlib.crc32(repr((key, self._version)).encode()))
            self._plans[key] = self.plan(start, goal, v0, rng=rng)
        return self._plans[key]

    def plan(self, start, goal, v0=None, rng=None):
        """Shortest feasible duration (grow N geometrically, then bisect down), then optional slack.

        With ``rng`` and ``duration_slack=(lo, hi)`` the plan is re-solved with
        N = N_min * U(lo, hi) waypoints, i.e. a slower trajectory with a varied speed level.
        """
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        v0 = np.zeros(3) if v0 is None else np.asarray(v0, float)
        limit = self.box_fraction * self.step_size
        n_lo = max(int(math.ceil(np.max(np.abs(goal - start)) / limit)) + 1, 2)
        n_hi = self.max_steps + 1
        if self._net is None or n_lo <= 2:
            n = n_lo
            if rng is not None and self.duration_slack is not None and n_lo > 2:
                n = min(int(round(n_lo * rng.uniform(*self.duration_slack))), n_hi)
            return self._as_plan(self._line(start, goal, n), True, n_lo - 1)
        failed, found = n_lo - 1, None
        for factor in self.growth:
            n = min(max(int(math.ceil(n_lo * factor)), failed + 1), n_hi)
            solution = self._solve(start, goal, v0, n)
            if solution is not None:
                found = (n, solution)
                break
            failed = n
            if n >= n_hi:
                break
        if found is None:
            self.failures += 1
            if self.verbose:
                print("planner: no feasible plan up to N=%d, using a straight line" % n_hi)
            return self._as_plan(self._line(start, goal, n_lo), False)
        for _ in range(self.bisect_iters):
            if found[0] - failed <= 1:
                break
            n = (failed + found[0]) // 2
            solution = self._solve(start, goal, v0, n)
            if solution is None:
                failed = n
            else:
                found = (n, solution)
        n_min = found[0]
        if rng is not None and self.duration_slack is not None:
            target = min(int(round(n_min * rng.uniform(*self.duration_slack))), n_hi)
            if target > n_min:
                solution = self._solve(start, goal, v0, target)
                if solution is not None:     # otherwise keep the fastest plan
                    found = (target, solution)
        if self.verbose:
            print("planner: N=%d (fastest feasible %d, lower bound %d)" % (found[0], n_min, n_lo))
        return self._as_plan(found[1], True, n_min - 1)

    def _as_plan(self, positions, feasible, min_steps=None):
        actions = np.diff(positions, axis=0) / self.step_size
        return Plan(positions, np.clip(actions, -1, 1), feasible,
                    len(actions) if min_steps is None else min_steps)

    @staticmethod
    def _line(start, goal, n):
        return start + (goal - start) * np.linspace(0, 1, n)[:, None]

    def _solve(self, start, goal, v0, n):
        """Try a straight and two curved initial guesses for a fixed N. Returns (N, 3) or None."""
        line = self._line(start, goal, n)
        side = np.array([-(goal - start)[1], (goal - start)[0], 0.0])
        side = side / np.linalg.norm(side) if np.linalg.norm(side) > 1e-9 else np.array([0, 1.0, 0])
        phase = np.sin(np.pi * np.linspace(0, 1, n))[:, None]
        for sign in (0.0, 1.0, -1.0):
            try:
                return self._solve_from(start, goal, v0, line + sign * self.bump * phase * side)
            except ValueError:
                continue
        return None

    def _solve_from(self, start, goal, v0, initial):
        """Penalty least squares on interior waypoints; see docs/nominal_planner.md."""
        n, m = len(initial), len(initial) - 2
        step, dt, I3 = self.step_size, self.dt, sparse.identity(3, format="csr")
        # P_ext = [x_-1, x_0, ..., x_N-1] = S z + b ; z are the interior waypoints
        S = sparse.lil_matrix((3 * (n + 1), 3 * m))
        for j in range(m):
            S[3 * (j + 2):3 * (j + 3), 3 * j:3 * (j + 1)] = np.eye(3)
        S = S.tocsr()
        b = np.zeros(3 * (n + 1))
        b[0:3], b[3:6], b[3 * n:3 * (n + 1)] = start - v0, start, goal
        diff1 = sparse.diags([-np.ones(n), np.ones(n)], [0, 1], shape=(n, n + 1))
        Dm = sparse.kron(diff1, I3, format="csr")                       # d_i = P_i - P_i-1, i = 0..N-1
        diff2 = sparse.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n))
        Em = sparse.kron(diff2, I3, format="csr")                       # acc_i = d_i+1 - d_i, i = 0..N-2
        Jd, cd = (Dm @ S).tocsr(), Dm @ b                               # d = Jd z + cd
        Ja, ca = (Em @ Jd).tocsr(), Em @ cd
        Jd1, cd1 = Jd[3:], cd[3:]                                       # drop d_0 = v0 (fixed)
        rows = sparse.vstack([math.sqrt(self.w_len) / step * Jd1,
                              math.sqrt(self.w_acc) / step * Ja], format="csr")
        offset = np.r_[math.sqrt(self.w_len) / step * cd1, math.sqrt(self.w_acc) / step * ca]
        objective = lambda z: (rows @ z + offset, rows)

        limit = self.box_fraction * step
        box_jac = sparse.vstack([-Jd1, Jd1], format="csr")
        constraints = [
            dict(scale=step, tolerance=0.01 * step, evaluate=lambda z: (
                np.r_[limit - (Jd1 @ z + cd1), limit + (Jd1 @ z + cd1)], box_jac)),
            dict(scale=step, tolerance=1e-6, evaluate=lambda z: (
                np.r_[z - np.tile(self.workspace_low, m), np.tile(self.workspace_high, m) - z],
                sparse.vstack([sparse.identity(3 * m), -sparse.identity(3 * m)], format="csr"))),
        ]
        if self._net is not None:
            constraints.append(dict(scale=0.1, tolerance=1e-3,
                                    evaluate=lambda z: self._learned(z, S, b, Jd, cd, goal, n)))
        if self.v_max is not None:
            c2 = (self.v_max * dt) ** 2
            constraints.append(dict(scale=c2, tolerance=1e-3 * c2, evaluate=lambda z: self._norm_cap(
                (Jd @ z + cd).reshape(-1, 3)[1:], Jd1, c2)))
        if self.a_max is not None:
            c2 = (self.a_max * dt * dt) ** 2
            constraints.append(dict(scale=c2, tolerance=1e-3 * c2, evaluate=lambda z: self._norm_cap(
                (Ja @ z + ca).reshape(-1, 3), Ja, c2)))
        z0 = initial[1:-1].reshape(-1)
        z = solve_penalty(z0, objective, constraints, max_nfev=self.max_evaluations,
                          weight=self.constraint_weight)
        return np.vstack([start, z.reshape(-1, 3), goal])

    @staticmethod
    def _norm_cap(vectors, jac, bound_sq):
        """value = bound^2 - |vector_i|^2 per row i, with Jacobian -2 vector_i . d vector_i/dz."""
        k = len(vectors)
        picker = sparse.csr_matrix((-2 * vectors.reshape(-1), (np.repeat(np.arange(k), 3), np.arange(3 * k))),
                                   shape=(k, 3 * k))
        return bound_sq - np.sum(vectors ** 2, axis=1), picker @ jac

    def _learned(self, z, S, b, Jd, cd, goal, n):
        """h_i = threshold - cost(obs_i) - margin >= 0 for i = 1..N-1, obs_i = [P_i, goal-P_i, P_i-P_i-1]."""
        ext = (S @ z + b).reshape(-1, 3)                                # ext[i+1] = P_i
        d = (Jd @ z + cd).reshape(-1, 3)
        idx = np.arange(1, n)
        obs = np.hstack([ext[idx + 1], goal - ext[idx + 1], d[idx]])
        cost, grad = net_cost_and_obs_grad(self._net, obs)
        value = self._net.manual_threshold - cost - self.margin
        g_now = grad[:, 0:3] - grad[:, 3:6] + grad[:, 6:9]              # d cost / d P_i
        g_prev = -grad[:, 6:9]                                          # d cost / d P_i-1
        k, cols = len(idx), np.arange(3)
        r = np.repeat(np.arange(k), 3)
        G = sparse.csr_matrix((np.r_[g_now.reshape(-1), g_prev.reshape(-1)],
                               (np.r_[r, r], np.r_[(3 * (idx[:, None] + 1) + cols).reshape(-1),
                                                   (3 * idx[:, None] + cols).reshape(-1)])),
                              shape=(k, 3 * (n + 1)))
        return value, -(G @ S)


class _Cursor:
    """Position inside a plan for one environment; follows the plan while the state matches it."""

    def __init__(self, plan, goal):
        self.plan, self.goal, self.index = plan, np.asarray(goal, float), 0

    def accepts(self, position, goal, tolerance=1e-4):
        if self.index >= len(self.plan.actions) or np.linalg.norm(goal - self.goal) > tolerance:
            return False
        return np.linalg.norm(position - self.plan.positions[self.index]) <= tolerance

    def next_action(self, position, step_size):
        if self.index < len(self.plan.actions):
            action = self.plan.actions[self.index]
        else:
            action = np.clip((self.goal - position) / step_size, -1, 1)
        self.index += 1
        return action
