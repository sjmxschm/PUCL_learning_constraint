# Running PUCL with velocity constraints (KukaCup1)

Two modes, chosen by `-cosd` (the network inputs). The scaling flags mean different things in each, so use the matching command.
Never pass `-dmwr` / `--ds_reference_point`: they only apply to `--planner ds`.
Add `-sf` only if a display is reachable (see "Plots"); otherwise PNGs are written to `icrl/runs/.../`.

## Mode A: position + velocity (`-cosd 0 1 2 6 7`)
Net inputs: x, y, z and the last displacement (obs 6:8, **m/step**, ~0.005).
```bash
python run_me.py dscl -g KukaCup1-velocity -tei KukaCup1-v0 -eei KukaCup1-v0 \
  -ep icrl/expert_data/KukaCup1 -er 10 -cosd 0 1 2 6 7 \
  -dno -dnr -dnc -rg 1 -cl 32 32 -clr 0.005 \
  -ni 10 -bi 300 -cbs 256 -een 10 -cpe 1 -twm -kNNt 0.03 \
  -cvs 125 -kNNsr <radius_m> -kNNvs <scale>
```
| Flag | Suggested | Meaning / effect if left out |
|---|---|---|
| `-cvs` | 125 (= 1/step_size) | Gain on velocity inputs inside the net. Without it the net sees ~0.005 and barely learns from velocity |
| `-kNNvs` | tune | Velocity multiplier in the kNN distance. Speed difference [m/s] counted as "far" = `kNNt / (scale·dt)`. Without it velocity is almost ignored (`-kNNt 0.03` vs. ~0.005) |
| `-kNNsr` | radius in m, tune (e.g. 0.05) | Slower-than-expert rule: a nominal point no faster than an expert within this position radius is not "reliably infeasible". Without it slow points near an expert can be labelled infeasible |

## Mode B: speed + distance to obstacle (`-cosd 11 12`)
Net inputs: speed |v| (**m/s**, ~0.2) and distance of the xy position to the obstacle border (m). Derived features, no raw position or velocity.
```bash
python run_me.py dscl -g KukaCup1-velocity -tei KukaCup1-v0 -eei KukaCup1-v0 \
  -ep icrl/expert_data/KukaCup1 -er 10 -cosd 11 12 \
  -dno -dnr -dnc -rg 1 -cl 32 32 -clr 0.005 \
  -ni 10 -bi 300 -cbs 256 -een 10 -cpe 1 -twm -kNNt 0.03 \
  -kNNvs 1
```
| Flag | Suggested | Meaning |
|---|---|---|
| `-cvs` | **leave out** (1) | Speed is already in m/s. `-cvs 125` would feed the net ~25 |
| `-kNNvs` | 1 as a starting point, tune | Speed multiplier in the kNN distance. Speed difference [m/s] counted as "far" = `kNNt / scale` (so 0.03 m/s at scale 1). Large values like 20 make it far too strict |
| `-kNNsr` | **do not use** | Raises `ValueError`: the rule needs raw position (dims 0-2) and velocity (dims 6-8), which mode B does not have |

Other allowed `-cosd` sets: `0 1 2`, `0 1 2 11`, `0 1 2 11 12`. Dims 11 is scaled like speed (m/s) in `-cvs` / `-kNNvs`; the raw-velocity rule `-kNNsr` only works with `0 1 2 6 7`.

## Both modes
| Flag | Value | Effect |
|---|---|---|
| `--planner` | `traj_opt` (default) or `ds` | Nominal trajectory generator |
| `--duration_slack LOW HIGH` | `1.0 2.5` (default); `1 1` = always fastest | Speed coverage of nominal data: plan length = fastest N · U(LOW, HIGH) |

## Constraints between flags
- `-kNNvs` cannot be combined with `-kNNn` (z-scoring undoes it; raises an error).
- Required for KukaCup1: `-dno -dnr -dnc`, no `-cn`, `-casd -1` (default), train and eval env both `KukaCup1-v0`.

## Plots
- Without `-sf`: PNGs only (Agg backend), including `velocity_<iter>.png` at 300 dpi. Works over ssh.
- With `-sf`: Tk windows that block the run until closed. Needs a display (`ssh -X` and `echo $DISPLAY` non-empty, or run locally).
- Mode B does not write the x-y constraint plot; only the velocity plot.

## Data
Expert rollouts must come from the time-based `icrl/tests/prepare_kuka_cup1_data.py` (default time resampling; `--retime` adds a synthetic speed profile that grows with distance to the obstacle; `--profile_gain`, `--v_min`, `--v_max`, `--a_max`). Arc-length (`position_only`) data has no real velocity information.

## Check
Δt comes from `task_config.json` (`sampling_time`, 0.04 s); the env default is 0.05. Make sure they agree.

See also: `velocity_position_pucl_summary.md`, `nominal_planner.md`.
