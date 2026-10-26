# PUCL with joint position + velocity constraints (KukaCup1)

## Idea
PUCL = positive-unlabeled constraint learning. Expert states are **positive** (feasible). Planner states are **unlabeled** (feasible or not). We pick the unlabeled points that are *surely* infeasible and train a classifier expert vs. those.

Ground-truth constraint (used only for evaluation): infeasible iff `speed > velocity_gain · dist_to_border`, gain = 1/s.
Inside the cylinder `dist < 0`, so the position constraint is a special case. The old explicit "inside the cylinder" check is now dead code (`KukaCup1.py` `true_cost_function`).

## Loop
1. **Data**: sub-optimal demos, resampled **in time** (0.04 s grid) instead of by arc length, so velocity exists. `icrl/tests/prepare_kuka_cup1_data.py` (`resample_by_time`, optional `--retime`).
2. **Plan** (`icrl/nominal_planner.py`, `Trajectory_Optimizer`, `--planner traj_opt`): per start/goal, least-squares trajectory optimization, minimizing path length and acceleration, subject to the *currently learned* constraint `θ − cost(o_i) ≥ margin` (gradient taken through the net) and the action box. Fixed Δt, so duration = number of waypoints N. Search the smallest feasible N, then re-solve with `N·U(1, 2.5)` (`--duration_slack`).
3. **Reliable infeasible** (`constraint_net.py`, `train_with_two_step_pu_learning`, kNN branch): nominal points whose kNN distance to the expert set is `> kNN_thresh`.
4. **Classify**: MLP + sigmoid, expert = 1, reliable infeasible = 0. Output is P(feasible), cost = 1 − output, infeasible if cost > 0.5.
5. Repeat until the constraint stops changing.

## What changed for velocity (and where)
| Change | Why | Where |
|---|---|---|
| Net input includes velocity: obs 6:8 = last displacement (m/step). Run with `-cosd 0 1 2 6 7`. Optional derived features 11 = speed (m/s), 12 = distance to border | The constraint depends on state *and* speed | `constraint_net.py` `select_appropriate_dims`; allowed dims checked in `dscl.py` |
| Fixed gain on velocity inputs inside the net (`cn_velocity_input_scale`, ≈1/step_size = 125) | Raw m/step values are ~0.005, so the net would barely see them | `constraint_net.py` `forward`, `_build` |
| kNN distance rescales velocity (`kNN_velocity_scale`) | Sets how big a speed difference at the same place counts as "far from experts" | `constraint_net.py`, kNN branch |
| **Slower-than-expert rule** (`kNN_slower_feasible_radius`) | The speed constraint is one-sided. A slow point near an expert is feasible, but plain kNN would call it infeasible because its velocity differs | `constraint_net.py`, right after the kNN threshold |
| True cost uses speed vs. border distance; reward adds a time penalty `−time_rate·dt` | Ground truth for velocity; shorter time is preferred | `gym_panda/envs/KukaCup1.py` `true_cost_function`, `step` |
| Planner replaces DS policy | The DS policy has no notion of time or speed | `icrl/nominal_planner.py`, selected in `dscl.py` |
| Velocity plot | Inspect the learned constraint in the (distance, speed) plane | `icrl/plot_velocity.py` |

## Open issue: velocity coverage
Minimum-time planning always picks the maximum speed, so the nominal data covers one speed level. Workaround: vary N per plan (`duration_slack`), which gives different speeds because Δt is fixed. Expert speeds (step 1) are also not known well enough. `--retime` assigns a synthetic speed profile (speed grows with distance to the obstacle), and the true timing is kept in `files/EXPERT_TIME`.

## Corrections to the Notion draft
- Net output is P(**feasible**), not P(constraint).
- The classifier trains on the **reliable infeasible subset** vs. experts, not on all nominal points.
- Δt is read from `task_config.json` (`sampling_time` 0.04 s). The env default is 0.05, so check these agree.
- The planner uses a *learned-constraint inequality*, not a hard CBF (the CBF is left out of v1; see `docs/nominal_planner.md` §3).
- `docs/nominal_planner.md` still says "not yet run end to end"; the last commit says it works.
