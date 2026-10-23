# Trajectory-optimization planner for DSCL (`icrl/nominal_planner.py`)

This replaces `DS_Policy` as the **nominal trajectory generator** in `dscl.py`. It is a
planner that sits behind the same two methods `dscl.py` already calls (`predict`,
`modulate_with_NN`). Nothing else in `dscl.py`, `utils.py` or the environment needs to change,
apart from the import (section 7).

Status: tested offline against an analytic stand-in for the constraint network and against the
real `ConstraintNet` gradient code. **Not yet run end to end inside `dscl.py`** (section 8).

---

## 1. Notation

| symbol | meaning | value / source |
|---|---|---|
| $\Delta t$ | environment time step | `env.dt` = 0.04 s (`task_config.json: sampling_time`) |
| $s$ | metres moved per unit action | `env.step_size` = 0.008 m |
| $N$ | number of waypoints $x_0,\dots,x_{N-1}$ (so $N-1$ environment steps, duration $(N-1)\Delta t$) | searched, see 3.6 and 3.7 |
| $x_0, g$ | start and goal | from the observation: $x_0=o[0{:}3]$, $g=o[0{:}3]+o[3{:}6]$ |
| $d_i = x_i-x_{i-1}$ | displacement of step $i$ (metres per step) | $d_0=v_0$ is the given previous displacement ($0$ at episode start) |
| $a_i = (d_{i+1}-d_i)/\Delta t^2$ | acceleration at $x_i$ | |
| $o_i=[\,x_i,\; g-x_i,\; d_i\,]\in\mathbb R^9$ | the observation the environment would give | same layout as `KukaCup1._observation` |
| $c(o)=1-\mathrm{net}(o)$ | learned constraint cost, $\in[0,1]$ | `ConstraintNet`; infeasible if $c>\theta$ |
| $\theta$ | threshold | `manual_threshold` = 0.5 |

Important: the network's velocity inputs (`-cosd ... 6 7`) are $d_i$, **metres per step**, not
m/s. Only `true_cost_function` divides by `dt`.

---

## 2. What `build_gamma_with_grad_for_ds` actually returns

From `constraint_net.py:190`:

$$
\texttt{out}=\mathrm{net}(x),\qquad c = 1-\texttt{out},\qquad
\texttt{grad}=\frac{\partial c}{\partial x},\qquad
\Gamma = 1+\max\big(10(\theta-c),\,0\big).
$$

Consequences:

* `grad` is the gradient of the **cost**, not of $\Gamma$, and only with respect to the
  **selected** inputs ($x$ is `prepare_data(obs, acs)` after `-cosd/-casd` selection).
* $\Gamma$ is **not a distance**. It is a piecewise-linear function of the network output:
  it equals exactly $1$ everywhere the state is infeasible ($c\ge\theta$) and saturates at
  $1+10\theta=6$ where $c=0$. Inside the unsafe set it is flat, so it carries no information
  about *how far inside* a point is.
* A usable barrier is the **unclipped** quantity $h(o)=\theta-c(o)$ ($h>0$ safe, $h=0$ boundary,
  $h<0$ unsafe). It is smooth, signed, and has a gradient on both sides. $\Gamma-1=10\max(h,0)$
  is just a clipped copy of it.

So the planner uses $h$ directly, which is why it needs the network object and not only the
returned `gamma`. `modulate_with_NN` gets the network from the bound method
(`gamma_function.__self__`).

---

## 3. The optimization problem that is solved

### 3.1 What changed relative to your formulation

| yours | implemented | why |
|---|---|---|
| free continuous $T$, $\Delta t=T/(N-1)$ | **fixed** $\Delta t=0.04$; $N$ is an integer found by search | The environment and the expert data live on the 0.04 s grid. A plan with another $\Delta t$ gives displacements (hence velocities) the environment cannot reproduce. $w_T T$ becomes "smallest feasible $N$". |
| $w_\ell\sum\|x_{i+1}-x_i\|^2$, $w_a\sum\|\ddot x\|^2$ | same, normalised by $s$ | dimensionless weights |
| $h_\theta(x_i)=0$ | $h(o_i)\ge\epsilon$ (inequality, see question Q1) | feasibility of the learned constraint is an inequality |
| CBF: $\Gamma(x_{i+1})-1\ge(1-\alpha\Delta t)(\Gamma(x_i)-1)$ | **not in v1** (see 3.4) | the network already sees velocity |
| $\|x_{i+1}-x_i\|\le v_{\max}\Delta t$ | optional (`v_max`) | |
| $\|\ddot x_i\|\le a_{\max}$ | optional (`a_max`) | |
| (none) | **box constraint** $\lvert d_{i,k}\rvert\le\beta s$ | `KukaCup1.step` clips every action component to $[-1,1]$. A plan outside this box cannot be executed. |
| (none) | workspace bounds | `step` clips to the workspace |

### 3.2 Problem for a fixed $N$

Variables: the interior waypoints $z=(x_1,\dots,x_{N-2})\in\mathbb R^{3(N-2)}$; $x_0$ and $x_{N-1}=g$ are fixed.

$$
\min_z\;\; w_\ell\sum_{i=1}^{N-1}\Big\|\tfrac{d_i}{s}\Big\|^2
\;+\; w_a\sum_{i=0}^{N-2}\Big\|\tfrac{d_{i+1}-d_i}{s}\Big\|^2
$$

subject to, for the indicated $i$:

$$
\begin{aligned}
&\text{(C1) box:} && |d_{i,k}|\le\beta s, && i=1..N-1,\;k\in\{x,y,z\}\\
&\text{(C2) workspace:} && x_{\text{low}}\le x_i\le x_{\text{high}} && i=1..N-2\\
&\text{(C3) learned:} && h(o_i)=\theta-c(o_i)-\epsilon\ \ge 0, && i=1..N-1\\
&\text{(C4) speed (optional):} && \|d_i\|^2\le (v_{\max}\Delta t)^2, && i=1..N-1\\
&\text{(C5) accel (optional):} && \|d_{i+1}-d_i\|^2\le (a_{\max}\Delta t^2)^2, && i=0..N-2
\end{aligned}
$$

with $\beta=0.97$ (so executed actions stay strictly inside $[-1,1]$), $\epsilon=0.02$ (a
margin so the *binary* learned cost is 0 on the plan), $w_\ell=1$, $w_a=5$.

Everything in the objective, C1, C2 is **linear** in $z$:
$d = J_d z + c_d$ and $d_{i+1}-d_i = J_a z + c_a$ with sparse constant matrices
(built from a difference operator and the fixed boundary points).
Only C3 (and C4/C5, which are quadratic) are nonlinear.

### 3.3 Gradient of C3 (chain rule)

`net_cost_and_obs_grad` returns $G=\partial c/\partial o\in\mathbb R^9$ (zero on unselected
entries). Because $o_i=[x_i,\,g-x_i,\,x_i-x_{i-1}]$:

$$
\frac{\partial c}{\partial x_i}=G_{0:3}-G_{3:6}+G_{6:9},\qquad
\frac{\partial c}{\partial x_{i-1}}=-G_{6:9},\qquad
\frac{\partial h}{\partial(\cdot)}=-\frac{\partial c}{\partial(\cdot)}.
$$

Each constraint row $i$ therefore touches only $x_i$ and $x_{i-1}$ (a banded sparse Jacobian).
The velocity dependence of the network is what couples neighbouring waypoints.

### 3.4 Where your control barrier function fits

For $h$ a distance to the obstacle, $\dot h\ge-\alpha h$ gives $v_\perp\le\alpha\,h$, i.e. exactly
the speed-versus-distance law of the true cost, $v\le\gamma\, d$, with $\alpha\leftrightarrow\gamma$
(`velocity_gain`). In discrete time that is your inequality
$h(x_{i+1})\ge(1-\alpha\Delta t)\,h(x_i)$.

Two reasons it is **not** in v1:

1. If the network is trained with velocity inputs ($-cosd\ 0\ 1\ 2\ 6\ 7$), the speed limit is
   already *inside* the learned feasible set $\{c<\theta\}$. C3 on $o_i$ enforces it. Adding
   a CBF on top would constrain the same thing twice.
2. $h=\theta-c$ is a classifier margin, not a metric distance, so $\alpha$ would not be in 1/s.

The CBF becomes useful if you learn a **position-only** network and want the speed law as an
explicit, interpretable layer on top. It would be one more linear-in-$h$ constraint
$h_{i+1}-(1-\alpha\Delta t)h_i\ge 0$ (needs $\partial h/\partial x_i$ at both waypoints, same
chain rule as 3.3).

### 3.5 How it is solved

`solve_penalty` (in `nominal_planner.py`) minimises
$\|r(z)\|^2+\sum_j (w/\sigma_j)^2\,\|\min(g_j(z),0)\|^2$ using `scipy.optimize.least_squares`
(trust-region, sparse LSMR) through `solve_trajectory_least_squares` from your new
`optimizer.py`. $w$ starts at 100 and is multiplied by 10 (up to 3 times) until every constraint is
within its tolerance; otherwise `ValueError` = "infeasible for this $N$".

### 3.6 What $N$ is, and how it is chosen

**$N$ is the number of waypoints in the plan, including start and goal.** The environment moves
one step per $\Delta t=0.04$ s, so a plan with $N$ waypoints is $N-1$ steps long and lasts
$(N-1)\Delta t$ seconds. $N=57$ means 56 steps, 2.2 s. $N$ is the *duration* of the trajectory.
Because the time grid is fixed, duration can only change in whole steps, so $N$ is an integer
you search over, not a continuous variable.

**Why "the smallest feasible $N$" first.** The environment's reward is
$-\text{path length}-\text{time\_rate}\cdot\Delta t$ per step. An agent that maximises it under the learned
constraint wants to arrive as quickly as the constraint allows. That is exactly your
$w_T\,T$ term. With $\Delta t$ fixed, "minimise $T$" means "smallest $N$ for which a feasible
path exists". The smallest feasible $N$ is the **time-optimal** plan for the current network. Without any
time pressure the smooth objective alone has no preference between $N=57$ and $N=200$.

How the smallest feasible $N$ is found:

* Lower bound from the action box: $N_{lo}=\lceil\max_k|g_k-x_{0,k}|/(\beta s)\rceil+1$ (the fastest
  straight line the actions allow, ignoring the network).
* Try $N=\lceil N_{lo}\cdot f\rceil$ for $f=1.1,1.25,1.5,2,3$ until feasible (capped at
  `max_steps+1`), then bisect between the last infeasible and the first feasible $N$ (3 steps).
  Feasibility is not guaranteed monotone in $N$, but in practice a longer horizon is easier.
* For each $N$ three starting paths are tried: straight line, and the line plus a lateral
  half-sine bump of $\pm0.08$ m. (A straight path through an obstacle centre has a
  zero-gradient symmetry; the bump breaks it. `optimizer.py`'s `neural_point_to_point` does the
  same thing.)
* No constraint network yet (iteration 0 of `dscl.py`) or $N_{lo}\le2$: straight line.
* If nothing is feasible: straight line, and `planner.failures` is incremented.

### 3.7 Speed variety: `duration_slack`

**Problem.** If every plan is time-optimal, the nominal trajectories all run at the action limit
(about 0.19 m/s per axis) wherever the network allows it. The velocity plot then shows a single
row of points at one speed. The experts are always slower than that, so the constraint network can
separate expert from nominal data with a *global speed cap* and never needs to learn that the
allowed speed grows with the distance to the obstacle ($v\le\gamma d$). The nominal data needs to
cover many speeds, at many distances.

**Why varied nominal data is legitimate.** The nominal data is *unlabeled* (the PU learning
setting): it is a mix of feasible and infeasible samples, and the network learns from the contrast
with the expert data. Slower plans are fine. Where they are feasible they look like the experts and
add no pressure there. Where a speed is infeasible, only the planner's *fast* plans expose it.
So variety does not hurt: slow plans anchor the feasible region at many speeds, fast plans probe
its boundary.

**What is implemented.** After the fastest feasible plan with $N_{\min}$ waypoints is found, the
planner draws $u\sim U(\text{lo},\text{hi})$ and **re-solves with $N=\mathrm{round}(N_{\min}\,u)$**
(capped at `max_steps+1`). Same objective, same constraints, more steps, so a lower speed level.
Default `duration_slack=(1.0, 2.5)`; `None` turns it off (always time-optimal).

* The draw is seeded by (start, goal, initial displacement, constraint version). It is
  reproducible, differs between demonstrations, and **changes every iteration** because the
  version changes whenever `modulate_with_NN` is called.
* The plan is cached, so `evaluate_goal_reaching_policy` replays the *same* plan that was sampled
  in that iteration. Evaluation metrics (reward, path length, episode length) therefore include
  the random slowdown. If you need the time-optimal numbers, set `duration_slack=None` for
  evaluation.
* If the re-solve at the larger $N$ fails, the fastest plan is kept.
* `Plan.min_steps` stores the fastest step count, so `len(plan.actions)/plan.min_steps` is the
  realised slowdown factor.
* Cost: one extra solve per plan (about 5 s per start instead of 2 to 3 s on the stand-in network).

**Trade-offs to keep in mind.**

* **Reward filter.** `select_policy_excerpts` only keeps a nominal trajectory if its return beats the
  expert's. A slow plan with $u=2.5$ may lose for the demonstrations that are already short. The
  expert returns are about $-0.62$ to $-0.99$, a typical fast plan about $-0.4$ to $-0.5$.
  If many rows print "Invalid unlabeled trajectory" at a high $u$, lower `hi`.
* **What this varies.** It changes the *overall speed level* between trajectories, through $N$.
  It does not by itself create "slow near the obstacle, fast far away" inside one trajectory; that
  comes from the network's constraint being active, which is what the data is meant to teach.
* **Smoothness weights.** At fixed $N$, $w_\ell$ and $w_a$ prefer an even speed. If you also want
  more variation *within* a trajectory, reduce $w_a$ or add random speed-profile warping (not implemented).

---

---

## 4. What already existed, and what I did with it

### `icrl/optimizer.py` (`StageConstraintTrajectoryOptimizer`, 1792 lines)

This comes from the robot-interface project (OptiTrack bar/table task frame, 7-D
position+quaternion poses, tool axes and yaw, staged constraints). The part that matters for
KukaCup1 is the `neural_point_to_point` backend:

* position-only path from start to goal, control points spaced **0.045 m in space** (not time),
* residuals: first-difference length vs. template, second-difference smoothness,
* learned constraint as a **soft penalty** $30\max(0.5-\text{score},0)$ at 9 samples per edge,
  with an analytic Jacobian,
* multi-start with $\pm0.08$ m lateral bumps.

Why it is not used directly:

1. `plan()` needs `NeuralSafetyConstraint` from `icrl/neural_constraint.py`, **which is not in the repo**.
2. It needs a large config (stage endpoints, constraint terms, task frame, bar/table geometry).
3. It has **no time axis** (spacing in metres), and its network wrapper scores positions only.
   So velocity cannot enter. That is the core requirement here.
4. No hard caps on speed or acceleration, only penalties.

Reused: `solve_trajectory_least_squares` (the shared `least_squares` settings).
Mirrored: smoothness via second differences, learned constraint as a one-sided penalty,
the lateral-bump initialisation.
I read `optimizer.py` but did not test it.

### `icrl/trajectory_optimization.py`
Was in the repo earlier and is gone now. `nominal_planner.py` does not depend on it.

### `DS_Policy` (the interface that is matched)

| `dscl.py` calls | `DS_Policy` | `Trajectory_Optimizer` |
|---|---|---|
| `Trajectory_Optimizer()` | | builds a default `KukaCup1()` to read `dt`, `step_size`, workspace, `max_steps` |
| `modulate_with_NN(gamma_fn, rho)` | builds a modulation matrix from `gamma_fn` | stores the network (`gamma_fn.__self__`), clears plan cache; `rho` ignored |
| `modulate_with_NN_with_refer_point(...)` | same, with reference point | same as above; reference point ignored |
| `predict(obs, state, deterministic)` | $\dot x$ for each row, norm $\le1$ | next action of the plan for each row |

`predict` is **open-loop replay with a safety net**: plans are cached by
(start, goal, initial displacement, constraint version); a row follows its plan while
$\|x_{\text{obs}}-x_{\text{plan}}\|\le10^{-4}$ m; if the state drifts or a new episode starts, it
re-plans from the current state.

---

## 5. Checks I ran

Offline, with an analytic stand-in network
$c=\sigma(-40(\gamma d-v))$, $\gamma=3$ (so "infeasible" $\Leftrightarrow v>\gamma d$), for the
first four demo start/goal pairs:

* All plans feasible; replayed through the real `KukaCup1`, all four **reached the goal with 0
  violations of the stand-in constraint**; maximum $|\text{action}|\approx0.98$.
* Planning takes 2 to 5 s per start/goal pair (about 5 s with `duration_slack`).
* Plans are 57 to 59 steps (2.3 s), much faster than the experts (109 to 221 steps),
  so their return is higher than the expert return (about $-0.5$ vs. $-0.8$). That is the
  condition `select_policy_excerpts` looks for.
* `duration_slack` (stand-in network, `demo_04` start/goal, four consecutive constraint versions):
  durations 88, 57, 106, 102 steps against a fastest plan of 56 steps (factors 1.02 to 1.89),
  top speed 0.22, 0.13, 0.11, 0.11 m/s. Every variant reached the goal with 0 violations of the stand-in constraint.
* `net_cost_and_obs_grad` matches `ConstraintNet.build_gamma_with_grad_for_ds`'s gradient on the
  selected dimensions and is zero elsewhere. With an untrained network (cost about 0.53 everywhere
  > threshold) the planner correctly reports "no feasible plan" and falls back to the straight line.

Expert check with the current `KukaCup1.true_cost_function` (`velocity_gain = 1`), on the
converted rollouts in `files/EXPERT`:

> **527 of 1732 expert observations (30%) violate the true cost.**
> The gain must be at least $\max_t \,(\text{speed}_t/d_t)\approx2.46\ \mathrm{s^{-1}}$ for the experts to be feasible.
> Experts stay at least 0.065 m from the cylinder border and reach 0.18 m/s.

So `velocity_gain=1` contradicts your demonstrations. Pick a value above about 2.5 (with margin)
or the "true" constraint you evaluate against is violated by the experts themselves.

---

## 6. Not verified

* A full `dscl.py` run (constraint training, the `-kNNt` policy-excerpt selection, evaluation).
* Behaviour with a *trained* network. The stand-in has a clean, smooth feasible set; a real one
  may have islands, in which case more starting paths or more evaluations (`max_evaluations`) may be needed.
* `optimizer.py` itself.
* Planning time at 10 demos per iteration: about 20 to 50 s per iteration if the numbers above hold.

---

## 7. Hooking it into `dscl.py` (your edits)

1. Import: `from icrl.nominal_planner import Trajectory_Optimizer`
   (`dscl.py:46` already calls `Trajectory_Optimizer()`; it is currently undefined).
2. Leave off `-dmwr` and `--ds_reference_point` (a planner has no reference point).
3. Same command otherwise, e.g.

```bash
python run_me.py dscl -g KukaCup1-velocity -tei KukaCup1-v0 -eei KukaCup1-v0 \
  -ep icrl/expert_data/KukaCup1 -er 10 -cosd 0 1 2 6 7 -dno -dnr -dnc -rg 1 \
  -cl 32 32 -clr 0.005 -ni 10 -bi 300 -cbs 256 -een 10 -cpe 1 -twm -kNNt 0.03
```

Tuning knobs are constructor arguments: `duration_slack` (default `(1.0, 2.5)`, `None` = always
time-optimal), `margin`, `w_len`, `w_acc`, `v_max`, `a_max`, `bump`, `max_evaluations`.
With `-sf` (`show_figures`) every plot opens a window and the run waits until you close it. Log `nominal_agent.failures` after sampling to see how many starts had no feasible plan.

---

## 8. Open questions for you

* **Q1.** What is $h_\theta(x_i)=0$ in your formulation: an equality that defines something, or
  the learned feasibility constraint written as an inequality? I implemented the inequality.
* **Q2.** Do you want the CBF inequality on top of a network that already takes velocity, or only
  if you move to a position-only network? What would you expect it to add?
* **Q3.** Which `velocity_gain` do you want, given the experts need at least 2.46?
* **Q4.** Is open-loop replay good enough, or should `predict` re-plan every step (model-predictive
  style)? What would that cost in planning time, and what would it fix?
