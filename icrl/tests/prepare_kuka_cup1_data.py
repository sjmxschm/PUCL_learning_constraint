"""Convert the selected raw iiwa7 segments into PUCL rollout dictionaries.

Run from the project root: python icrl/tests/prepare_kuka_cup1_data.py
Actions are Cartesian displacements divided by the simulated dynamics step,
not recorded robot commands. Original joint angles and poses remain in raw/.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline


def resample_by_arc_length(positions, times, num_points):
    """Interpolate spatially equally spaced positions while retaining both endpoints."""
    if num_points < 2:
        raise ValueError('num_points must be at least 2')
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
    unique_arc, indices = np.unique(arc, return_index=True)
    if len(unique_arc) < 2:
        raise ValueError('cannot resample a stationary trajectory')
    targets = np.linspace(0.0, unique_arc[-1], num_points)
    sampled = np.column_stack([np.interp(targets, unique_arc, positions[indices, dimension])
                               for dimension in range(3)])
    sampled_times = np.interp(targets, unique_arc, times[indices])
    sampled[0], sampled[-1] = positions[0], positions[-1]
    sampled_times[0], sampled_times[-1] = times[0], times[-1]
    return sampled, sampled_times

def resample_by_time(positions, times,
                     fs: int = 200, sample_reduction: int = 8) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Interpolate timewise equally spaced positions while retaining both endpoints.

        Args
        - positions: (N, d) vector of N positions in d dimensions
        - times: (N,) vector of N time steps
        - fs: sampling frequency in Hz
        - sample_reduction: factor of how much the sampling should be reduced

        Outpus:
        - sampled_positions: (N * fs/sample_reduction, d) vector of N positions in d dimensions
        - sampled_times: (N * fs/sample_reduction,) vector of N time steps
        - sample_reduction * ts: resampled sampling time
    """
    ts = 1. / fs  # true sampling time

    # times: (N,), position: (N, d)
    idx = np.arange(len(positions) - 1, -1, -sample_reduction)[::-1]   # stride anchored at the last sample
    sampled_times = times[idx]
    sampled_positions = positions[idx]

    step = np.diff(sampled_times)
    assert np.max(np.abs(step - sample_reduction * ts)) < 0.15 * sample_reduction * ts

    return sampled_positions, sampled_times, sample_reduction*ts

def retime_by_obstacle_distance(positions, center, radius, profile_gain=1.0, v_min=0.02, v_max=0.19, a_max=1.5):
    """New timestamps for the SAME path: slow near the obstacle, fast far from it.

    The geometry is untouched. Each raw sample gets a target speed
    ``clip(profile_gain * d, v_min, v_max)`` with ``d`` the planar distance to the cylinder border
    (so the speed rises with distance, v ~ profile_gain * d, like the true limit). A forward and a
    backward pass then cap the acceleration at ``a_max`` (m/s^2), starting from and ending at ``v_min``,
    so the trajectory speeds up after the start instead of jumping. Time is integrated as
    ``dt_i = ds_i / mean speed of the segment``.

    Args:
        positions: (N, 3) raw path in metres.
        center, radius: cylinder centre (x, y) and radius in metres.
        profile_gain: 1/s, slope of the speed profile against the distance to the border.
        v_min, v_max: speed floor and ceiling in m/s (v_max stays below the action limit, about 0.19 m/s).
        a_max: acceleration limit in m/s^2.

    Returns:
        (N,) timestamps in seconds, starting at 0 and non-decreasing.
    """
    positions = np.asarray(positions, dtype=float)
    ds = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    distance = np.linalg.norm(positions[:, :2] - np.asarray(center, dtype=float), axis=1) - radius
    speed = np.clip(profile_gain * distance, v_min, v_max)
    speed[0] = min(speed[0], v_min)
    for i in range(1, len(speed)):                      # forward: limited acceleration
        speed[i] = min(speed[i], np.sqrt(speed[i - 1] ** 2 + 2 * a_max * ds[i - 1]))
    speed[-1] = min(speed[-1], v_min)
    for i in range(len(speed) - 2, -1, -1):             # backward: limited deceleration
        speed[i] = min(speed[i], np.sqrt(speed[i + 1] ** 2 + 2 * a_max * ds[i]))
    segment_speed = np.maximum((speed[:-1] + speed[1:]) / 2, 1e-6)
    return np.r_[0.0, np.cumsum(ds / segment_speed)]


def resample_to_grid(positions, times, sampling_time):
    """Linear interpolation onto a uniform time grid anchored at the last sample (keeps the terminal point)."""
    steps = int(np.floor((times[-1] - times[0]) / sampling_time))
    grid = times[-1] - sampling_time * np.arange(steps, -1, -1)
    sampled = np.column_stack([np.interp(grid, times, positions[:, k]) for k in range(positions.shape[1])])
    return sampled, grid


def prepare(dataset_dir=None, num_points=100, position_only: bool = False,
            time_rate: float = 0.05, retime: bool = False, profile_gain: float = 1.0,
            v_min: float = 0.02, v_max: float = 0.19, a_max: float = 1.5):
    """
        position_only: specifies if only a positional constraint should be learned. If so,
            the trajectories are getting resampled by arc-length. If not, resampled by time.
        retime: re-index the raw timestamps with ``retime_by_obstacle_distance`` (same path, speed grows with
            the distance to the obstacle) before resampling onto the environment's time grid.
    """
    root = Path(dataset_dir) if dataset_dir else Path(__file__).resolve().parents[1] / "expert_data/KukaCup1"
    manifest = json.loads((root / "manifest.json").read_text())
    config_file = root / "task_config.json"
    config = json.loads(config_file.read_text()) if config_file.exists() else {
        "coordinate_frame": "iiwa7_link_0",
        "units": "metres",
        "obstacle_center": [0.60, -0.08],
        "obstacle_radius": 0.04,
        "geometry_source": "Estimated evaluation cylinder, not measured geometry.",
        "workspace_low": [0.3, -0.4, 0.1],
        "workspace_high": [0.9, 0.3, 0.6],
        "step_size": 0.008,
        "goal_tolerance": 0.01,
        "max_steps": 250,
        "time_rate": time_rate,
        "velocity_gain": 1.0,
    }
    step_size = config["step_size"]
    time_rate = config.get("time_rate", time_rate)
    tasks_so_far = config.get("tasks") or [{}]
    grid_time = tasks_so_far[0].get("sampling_time", 0.04)
    rollouts = root / "files/EXPERT/rollouts"
    rollouts.mkdir(parents=True, exist_ok=True)
    tasks = []
    for index, demo in enumerate(manifest["selected_demo_ids"]):
        with np.load(root / "raw" / (demo + ".npz"), allow_pickle=False) as data:
            positions = data["poses"][:, :3].copy()
            times = data["t"].copy()
        goal = positions[-1].copy()
        # End when the simulation would first satisfy its goal tolerance.
        hits = np.flatnonzero(np.linalg.norm(positions - goal, axis=1) <= config["goal_tolerance"])
        stop = int(hits[0])
        if stop < 1:
            raise ValueError(f"{demo}: start is already at the goal")
        positions, times = positions[:stop + 1], times[:stop + 1]
        source_num_positions = len(positions)
        source_path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
        if position_only:
            positions, times = resample_by_arc_length(positions, times, num_points)
        elif retime:
            times = retime_by_obstacle_distance(positions, config["obstacle_center"], config["obstacle_radius"],
                                                profile_gain, v_min, v_max, a_max)
            sampling_time = grid_time
            positions, times = resample_to_grid(positions, times, sampling_time)
            if len(positions) - 1 > config["max_steps"]:
                raise ValueError(f"{demo}: {len(positions) - 1} transitions exceed max_steps")
        else:
            positions, times, sampling_time = resample_by_time(positions, times)
            if len(positions) - 1 > config["max_steps"]:
                raise ValueError(f"{demo}: {len(positions) - 1} transitions exceed max_steps")
        displacements = np.diff(positions, axis=0)
        actions = displacements / step_size
        if np.any(np.abs(actions) > 1.0 + 1e-6):
            raise ValueError(f"{demo}: source transitions exceed action bounds; resample explicitly")
        if np.any(positions < config["workspace_low"]) or np.any(positions > config["workspace_high"]):
            raise ValueError(f"{demo}: source positions exceed the configured workspace")
        previous_displacements = np.vstack([np.zeros(3), displacements[:-1]])
        observations = np.concatenate([positions[:-1], goal - positions[:-1], previous_displacements], axis=1)
        path_length = float(np.linalg.norm(displacements, axis=1).sum())
        task = {"demo_id": demo, "rollout_file": f"{index}.pkl", "start": positions[0].tolist(),
                "goal": goal.tolist(), "terminal_position": positions[-1].tolist(),
                "num_transitions": len(actions), "num_positions": len(positions),
                "source_num_positions": source_num_positions,
                "source_path_length": source_path_length,
                "path_length": path_length}
        rollout = {
            "observations": observations.astype(np.float32),
            "actions": actions.astype(np.float32),
            "rewards": np.array([-path_length], dtype=np.float64),
            "lengths": np.array([len(actions)], dtype=np.int64),
            "save_scheme": "EXPERT",
            "source_demo": demo,
            "goal": goal,
            "terminal_observation": np.concatenate([positions[-1], goal - positions[-1], displacements[-1]]).astype(np.float32),
            "timestamps": times,
            "action_definition": "delta_position / step_size; derived, not measured robot command",
            "downsampling": {"method": "uniform_arc_length", "num_positions": num_points},
        }
        if not position_only:
            task["sampling_time"] = sampling_time
            rollout["rewards"] = np.array([-path_length -
                                           time_rate*(times[-1] - times[0])], dtype=np.float64)
            rollout["downsampling"] = {"method": "time_based", "sampling_time": sampling_time}
            if retime:
                rollout["downsampling"] = {"method": "retimed_by_obstacle_distance", "sampling_time": sampling_time,
                                           "profile_gain": profile_gain, "v_min": v_min, "v_max": v_max, "a_max": a_max}

        with (rollouts / task["rollout_file"]).open("wb") as stream:
            pickle.dump(rollout, stream, protocol=4)
        tasks.append(task)
    config["tasks"] = tasks
    config_file.write_text(json.dumps(config, indent=2) + "\n")
    manifest["conversion_status"] = "pucl_rollouts_ready"
    manifest["converted_rollouts"] = tasks
    manifest["downsampling"] = {"method": "uniform_arc_length", "num_positions_per_demo": num_points}
    if retime:
        manifest["downsampling"] = {"method": "retimed_by_obstacle_distance", "sampling_time": grid_time,
                                    "profile_gain": profile_gain, "v_min": v_min, "v_max": v_max, "a_max": a_max,
                                    "note": "synthetic timing on the true paths; true timing is in files/EXPERT_TIME"}
    manifest["total_training_observations"] = sum(task['num_transitions'] for task in tasks)
    manifest["notes"] = [
        "raw/ arrays are unchanged; converted rollouts use Cartesian position only.",
        "Goals are individual demonstration endpoints. Each converted trajectory stops at its first goal-tolerance hit.",
        (f"Converted trajectories are re-timed (speed grows with the distance to the obstacle, profile_gain={profile_gain}, "
         f"v in [{v_min}, {v_max}] m/s, a_max={a_max} m/s^2) and resampled every {grid_time} s, anchored at the terminal point. "
         "The true timing is kept in files/EXPERT_TIME." if retime else
         f"Converted trajectories are resampled to {num_points} positions by uniform arc length, including both endpoints."),
        "Actions are derived from displacement / step_size; no measured action or reward exists in the source.",
        ("Rewards are minus path length minus time_rate times duration." if (retime or not position_only) else
         "Rewards are negative accumulated Cartesian path length in metres."),
        "The default evaluation cylinder is estimated; its geometry is not used as training labels.",
    ]
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--points', type=int, default=100, help='number of positions including the terminal point')
    parser.add_argument('--retime', action='store_true', help='re-index time: slow near the obstacle, fast far away')
    parser.add_argument('--profile_gain', type=float, default=1.0, help='1/s, speed = gain * distance to the border')
    parser.add_argument('--v_min', type=float, default=0.02)
    parser.add_argument('--v_max', type=float, default=0.19)
    parser.add_argument('--a_max', type=float, default=1.5, help='acceleration limit, m/s^2')
    args = parser.parse_args()
    tasks = prepare(num_points=args.points, retime=args.retime, profile_gain=args.profile_gain,
                    v_min=args.v_min, v_max=args.v_max, a_max=args.a_max)
    print(f"Prepared {len(tasks)} KukaCup1 expert rollouts")
    for task in tasks:
        print(task["demo_id"], task["rollout_file"], task["num_transitions"], task["path_length"])
