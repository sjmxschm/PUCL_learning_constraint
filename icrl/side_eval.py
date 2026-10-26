"""Side evaluation for the real-robot pipeline: a hand-labeled constraint, so learning is not the unknown.

The true constraint is written down analytically (infinite cylinder with a margin, plus a speed limit that is
zero on the margin border and grows with the distance to it, capped by what the robot can do). Points are
sampled over the part of the state space the demonstrations live in, labeled with that rule, and a plain
classifier is trained on them with ``ConstraintNet`` and saved with ``ConstraintNet.save`` -- the same .pt format
DSCL writes. If the robot behaves correctly with this file, later problems come from the learning, not the robot side.

True rule (d = distance of the xy position to the cylinder border, v = planar speed in m/s)::

    position infeasible:  d < margin
    speed limit:          v_lim(d) = min(gain * (d - margin), v_cap)          (0 on the border, then rising)
    infeasible:           d < margin  or  v > v_lim(d)

Usage::

    python -m icrl.side_eval train  --out icrl/side_eval/run1
    python -m icrl.side_eval simulate --out icrl/side_eval/run1
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch as th

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

EXPERT_PATH = os.path.join(ROOT, 'icrl/expert_data/KukaCup1')
DT = 0.04                      # only used to put the speed into obs[6:8] (m/step); the network input is m/s
ROBOT_ROOT = os.path.join(os.path.dirname(ROOT), 'constraint_learning_robot_interface')
ROBOT_SETTINGS = os.path.join(ROBOT_ROOT, 'data/runtime/iiwa7_obsavoid/gui/obsavoid/settings.json')
ROBOT_TASK = os.path.join(ROBOT_ROOT, 'config/setups/iiwa7_obsavoid/tasks/obsavoid.json')


# ---------------------------------------------------------------------------------------------- the true rule
class TrueRule:
    """Analytic ground truth. ``distance`` is measured from the cylinder border, ``margin`` is added to it."""

    def __init__(self, center, radius, margin, gain, v_cap):
        self.center, self.radius, self.margin = np.asarray(center, float), float(radius), float(margin)
        self.gain, self.v_cap = float(gain), float(v_cap)

    def distance(self, xy):
        return np.linalg.norm(np.asarray(xy)[..., :2] - self.center, axis=-1) - self.radius

    def speed_limit(self, d):
        return np.clip(self.gain * (np.asarray(d) - self.margin), 0.0, self.v_cap)

    def position_feasible(self, xy):
        return self.distance(xy) >= self.margin

    def feasible(self, xy, speed):
        d = self.distance(xy)
        return (d >= self.margin) & (np.asarray(speed) <= self.speed_limit(d))

    def to_dict(self):
        return dict(center=self.center.tolist(), radius=self.radius, margin=self.margin, gain=self.gain, v_cap=self.v_cap)


# ---------------------------------------------------------------------------------------------- state-space region
def load_demo_positions(expert_path=EXPERT_PATH, load_type='EXPERT'):
    from icrl import utils
    (obs, _, _), lengths, _ = utils.load_expert_data(expert_path, 1000, load_type=load_type)
    return np.asarray(obs)[:, :3], lengths


def robot_endpoints():
    """GUI start/goal of the robot task, so that the sampled region contains what the robot will really do."""
    return np.array([[pose[k] for k in 'xyz'] for pose in robot_poses()]) if robot_poses() else np.zeros((0, 3))


def robot_poses():
    """[start, goal] poses (x y z qx qy qz qw dicts) from the robot GUI settings, else the task defaults."""
    if os.path.exists(ROBOT_SETTINGS):
        settings = json.load(open(ROBOT_SETTINGS))
        return [settings['start_by_task']['obsavoid'], settings['goal_by_task']['obsavoid']]
    if os.path.exists(ROBOT_TASK):
        gui = json.load(open(ROBOT_TASK))['gui']
        return [gui['default_start'], gui['default_goal']]
    return []


def sampling_box(demo_positions, rule, extra_points, pad_xy, pad_z):
    """Bounding box of the demonstrations, the cylinder (+ margin + room to accelerate) and extra points."""
    points = np.vstack([demo_positions, extra_points]) if len(extra_points) else demo_positions
    low, high = points.min(0) - [pad_xy, pad_xy, pad_z], points.max(0) + [pad_xy, pad_xy, pad_z]
    reach = rule.radius + rule.margin + rule.v_cap / rule.gain + 0.03
    low[:2] = np.minimum(low[:2], rule.center - reach)
    high[:2] = np.maximum(high[:2], rule.center + reach)
    return low, high


# ---------------------------------------------------------------------------------------------- sampling
def sample_points(n_uniform, n_boundary, box, rule, speed_max, rng):
    """Uniform samples over (x, y, z, v) plus samples concentrated at both boundaries.

    Boundary samples sit close to the margin ring (position boundary) or close to the speed-limit curve
    v_lim(d) (velocity boundary), on both sides, so the network sees where the label flips.
    """
    low, high = box
    uniform = np.column_stack([rng.uniform(low[i], high[i], n_uniform) for i in range(3)] + [rng.uniform(0, speed_max, n_uniform)])

    n_ring, n_speed = n_boundary // 2, n_boundary - n_boundary // 2
    angle = rng.uniform(0, 2 * np.pi, n_boundary)
    d_ring = rule.margin + rng.normal(0.0, 0.012, n_ring)                            # position boundary
    d_ramp = rule.margin + rng.uniform(-0.01, rule.v_cap / rule.gain + 0.03, n_speed)  # along the speed ramp
    d = np.concatenate([d_ring, d_ramp])
    xy = rule.center + (rule.radius + d)[:, None] * np.column_stack([np.cos(angle), np.sin(angle)])
    z = rng.uniform(low[2], high[2], n_boundary)
    limit = rule.speed_limit(d)
    v = np.concatenate([rng.uniform(0, speed_max, n_ring),
                        np.clip(limit[n_ring:] + rng.normal(0.0, 0.02, n_speed), 0, speed_max)])   # near v_lim(d)
    boundary = np.column_stack([xy, z, v])
    boundary = boundary[np.all(boundary[:, :3] >= low, axis=1) & np.all(boundary[:, :3] <= high, axis=1)]
    return np.vstack([uniform, boundary])


def to_observations(points, rng):
    """(x, y, z, v) -> 9-D observation. obs[6:8] holds a displacement of |v|*DT in a random planar direction,
    so ``ConstraintNet.select_appropriate_dims`` recomputes exactly v for feature 11."""
    phi = rng.uniform(0, 2 * np.pi, len(points))
    obs = np.zeros((len(points), 9))
    obs[:, :3] = points[:, :3]
    obs[:, 6] = points[:, 3] * DT * np.cos(phi)
    obs[:, 7] = points[:, 3] * DT * np.sin(phi)
    return obs


# ---------------------------------------------------------------------------------------------- training
def make_net(select_dim, hidden, rule, seed):
    from icrl.constraint_net import ConstraintNet
    th.manual_seed(seed)
    return ConstraintNet(9, 3, list(hidden), None, None, optimizer_class=None, is_discrete=False,
                         obs_select_dim=list(select_dim), acs_select_dim=[-1], sampling_time=DT,
                         obstacle_info={'center': rule.center.tolist(), 'radius': rule.radius}, device='cpu')


def features(net, obs):
    return net.select_appropriate_dims(th.as_tensor(np.concatenate([obs, np.zeros((len(obs), 3))], axis=-1), dtype=th.float32))


def train_classifier(net, obs, labels, epochs, lr, batch, seed, log=print):
    """BCE on standardised inputs; the standardisation is folded into the first Linear afterwards, so the saved
    network takes the raw inputs (metres, m/s) -- the robot rebuilds a plain Linear/LeakyReLU MLP and applies nothing else."""
    x, y = features(net, obs), th.as_tensor(labels, dtype=th.float32)[:, None]
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    layers = net.network
    optimizer = th.optim.Adam(layers.parameters(), lr=lr)
    steps = epochs * int(np.ceil(len(x) / batch))
    schedule = th.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps)
    generator = th.Generator().manual_seed(seed)
    for epoch in range(epochs):
        order = th.randperm(len(x), generator=generator)
        total = 0.0
        for i in range(0, len(x), batch):
            idx = order[i:i + batch]
            loss = th.nn.functional.binary_cross_entropy_with_logits(layers[:-1]((x[idx] - mean) / std), y[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step(); schedule.step()
            total += loss.item() * len(idx)
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            log('  epoch %3d  bce %.4f' % (epoch, total / len(x)))
    with th.no_grad():                                       # fold (x - mean) / std into layer 0
        first = layers[0]
        first.bias.copy_(first.bias - (first.weight * (mean / std)).sum(1))
        first.weight.copy_(first.weight / std)
        folded = layers(x)
        reference = th.sigmoid(layers[:-1](x))
        assert (folded - reference).abs().max() < 1e-5
    return net


def predict_feasible(net, obs):
    with th.no_grad():
        return net(features(net, obs)).numpy()[:, 0]


def classification_report(net, obs, truth, rule, use_speed, border=None):
    """Overall and boundary-region error rates. False feasible = the dangerous direction."""
    score = predict_feasible(net, obs)
    pred = score >= 0.5
    d = rule.distance(obs[:, :2])
    speed = np.linalg.norm(obs[:, 6:8], axis=1) / DT
    gap = np.minimum(np.abs(d - (rule.margin if border is None else border)), np.abs(speed - rule.speed_limit(d)) if use_speed else np.inf)
    out = {}
    for name, mask in (('all', np.ones(len(obs), bool)), ('near_boundary_1.5cm_or_0.02ms', gap < (0.015 if not use_speed else 0.02))):
        t, p = truth[mask], pred[mask]
        out[name] = dict(n=int(mask.sum()), accuracy=float((t == p).mean()),
                         false_feasible=float((~t & p).sum() / max((~t).sum(), 1)),
                         false_infeasible=float((t & ~p).sum() / max(t.sum(), 1)))
    return out


def cmd_train(args):
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    task = json.load(open(os.path.join(EXPERT_PATH, 'task_config.json')))
    rule = TrueRule(task['obstacle_center'], task['obstacle_radius'], args.margin, args.gain, args.v_cap)
    demos, lengths = load_demo_positions(load_type=args.expert_type)
    box = sampling_box(demos, rule, robot_endpoints(), args.pad_xy, args.pad_z)
    speed_max = args.speed_range * rule.v_cap
    print('region  low %s  high %s  | speed 0..%.3f m/s (v_cap %.3f)' % (np.round(box[0], 3), np.round(box[1], 3), speed_max, rule.v_cap))

    rng = np.random.RandomState(args.seed)
    train_pts = sample_points(args.n_uniform, args.n_boundary, box, rule, speed_max, rng)
    test_pts = sample_points(args.n_uniform // 4, args.n_boundary // 4, box, rule, speed_max, np.random.RandomState(args.seed + 1))
    train_obs, test_obs = to_observations(train_pts, rng), to_observations(test_pts, rng)
    labels = {'position_speed': rule.feasible(train_pts[:, :2], train_pts[:, 3]),
              'position': rule.position_feasible(train_pts[:, :2])}
    truth = {'position_speed': rule.feasible(test_pts[:, :2], test_pts[:, 3]),
             'position': rule.position_feasible(test_pts[:, :2])}
    np.savez_compressed(os.path.join(out, 'dataset.npz'), train_points=train_pts, test_points=test_pts,
                        train_label_position=labels['position'], train_label_position_speed=labels['position_speed'],
                        test_label_position=truth['position'], test_label_position_speed=truth['position_speed'],
                        box_low=box[0], box_high=box[1], demo_positions=demos)
    print('train %d points, feasible fraction: position %.2f, position+speed %.2f' % (
        len(train_pts), labels['position'].mean(), labels['position_speed'].mean()))

    inflate = args.task_speed / rule.gain      # border at which a constant task_speed is allowed by the true speed limit
    labels['position_inflated'] = rule.distance(train_pts[:, :2]) >= rule.margin + inflate
    truth['position_inflated'] = rule.distance(test_pts[:, :2]) >= rule.margin + inflate
    variants = {'position_speed': args.select_dim, 'position': [0, 1, 2], 'position_inflated': [0, 1, 2]}
    report = {'rule': rule.to_dict(), 'inflate_m': inflate, 'task_speed': args.task_speed, 'box_low': box[0].tolist(), 'box_high': box[1].tolist(), 'speed_max_sampled': speed_max,
              'hidden_sizes': args.hidden, 'seed': args.seed, 'files': {}, 'metrics': {}}
    for name, select_dim in variants.items():
        print('training %s  (-cosd %s)' % (name, ' '.join(map(str, select_dim))))
        net = make_net(select_dim, args.hidden, rule, args.seed)
        train_classifier(net, train_obs, labels[name], args.epochs, args.lr, args.batch, args.seed)
        path = os.path.join(out, 'side_eval_%s.pt' % name)
        net.save(path)
        report['files'][name] = dict(path=path, select_dim=list(select_dim))
        report['metrics'][name] = classification_report(net, test_obs, truth[name], rule, use_speed=11 in select_dim, border=rule.margin + (inflate if name == 'position_inflated' else 0.0))
        print(json.dumps(report['metrics'][name], indent=1))
    json.dump(report, open(os.path.join(out, 'config.json'), 'w'), indent=2)
    plot_dataset(out)
    plot_learned(out)


# ---------------------------------------------------------------------------------------------- loading helpers
def load_run(out):
    from icrl.constraint_net import ConstraintNet
    config = json.load(open(os.path.join(out, 'config.json')))
    rule = TrueRule(**config['rule'])
    data = np.load(os.path.join(out, 'dataset.npz'))
    nets = {k: ConstraintNet.load(v['path']) for k, v in config['files'].items()}
    return config, rule, data, nets


def plot_dataset(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    config, rule, data, _ = load_run(out)
    pts, lab_pos, lab_all = data['train_points'], data['train_label_position'], data['train_label_position_speed']
    box_low, box_high, demos = data['box_low'], data['box_high'], data['demo_positions']
    sub = np.random.RandomState(0).choice(len(pts), min(len(pts), 12000), replace=False)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))

    ax = axes[0]
    ax.scatter(pts[sub, 0], pts[sub, 1], c=np.where(lab_pos[sub], 'tab:green', 'tab:red'), s=2, alpha=.4)
    ax.add_patch(Circle(rule.center, rule.radius, fill=False, ec='k', lw=2, label='cylinder (r=%.0f mm)' % (rule.radius * 1e3)))
    ax.add_patch(Circle(rule.center, rule.radius + rule.margin, fill=False, ec='k', ls='--', label='+ margin %.0f mm' % (rule.margin * 1e3)))
    ax.plot(demos[:, 0], demos[:, 1], '.', color='navy', ms=2, label='demonstrations')
    ax.add_patch(plt.Rectangle(box_low[:2], *(box_high - box_low)[:2], fill=False, ec='grey', ls=':', label='sampled region'))
    ax.set(aspect='equal', xlabel='x (m)', ylabel='y (m)', title='position labels (top view)')
    ax.legend(loc='upper left', fontsize=8)

    d, v = rule.distance(pts[:, :2]), pts[:, 3]
    for ax, lab, title in ((axes[1], lab_all, 'position + speed labels  (d, v)'), (axes[2], lab_pos, 'position-only labels  (d, v)')):
        ax.scatter(d[sub], v[sub], c=np.where(lab[sub], 'tab:green', 'tab:red'), s=2, alpha=.4)
        dd = np.linspace(0, 0.3, 300)
        ax.plot(dd, rule.speed_limit(dd) * (dd >= rule.margin), 'k', lw=2, label='true limit min(gain (d - margin), v_cap)')
        ax.axvline(rule.margin, color='k', ls='--', lw=1, label='margin border')
        ax.axhline(rule.v_cap, color='tab:blue', ls=':', label='v_cap = %.2f m/s' % rule.v_cap)
        ax.set(xlim=(-0.04, 0.3), ylim=(0, v.max() * 1.02), xlabel='distance to cylinder border d (m)', ylabel='planar speed v (m/s)', title=title)
        ax.legend(loc='upper left', fontsize=8)
    fig.suptitle('Synthetic training set: green = feasible, red = infeasible (%d points)' % len(pts))
    fig.savefig(os.path.join(out, 'dataset.png'), bbox_inches='tight', dpi=130)
    plt.close(fig)


def plot_learned(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    config, rule, data, nets = load_run(out)
    box_low, box_high, demos = data['box_low'], data['box_high'], data['demo_positions']
    z0 = float(np.median(demos[:, 2]))
    rng = np.random.RandomState(0)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))

    # (1) position map at several speeds
    ax = axes[0]
    gx, gy = np.meshgrid(np.linspace(box_low[0], box_high[0], 160), np.linspace(box_low[1], box_high[1], 160))
    xyz = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z0)])
    net = nets['position']
    score = predict_feasible(net, to_observations(np.column_stack([xyz, np.zeros(len(xyz))]), rng)).reshape(gx.shape)
    ax.contourf(gx, gy, score, levels=[0, .5, 1], colors=['#f6c6c6', '#c9ecc9'])
    ax.add_patch(Circle(rule.center, rule.radius + rule.margin, fill=False, ec='k', ls='--'))
    ax.add_patch(Circle(rule.center, rule.radius, fill=False, ec='k'))
    ax.plot(demos[:, 0], demos[:, 1], '.', color='navy', ms=1.5)
    ax.set(aspect='equal', xlabel='x (m)', ylabel='y (m)', title='position-only network (z=%.2f): green = feasible\ndashed = true margin border' % z0)

    # (2) learned speed limit versus true limit, along a ray away from the cylinder
    ax = axes[1]
    angle = 0.0
    for name, color in (('position_speed', 'tab:blue'),):
        net = nets[name]
        dd = np.linspace(0, 0.3, 200)
        speeds = np.linspace(0, 1.3 * rule.v_cap, 130)
        for angle in np.linspace(0, 2 * np.pi, 5)[:-1]:
            xy = rule.center + (rule.radius + dd)[:, None] * np.array([np.cos(angle), np.sin(angle)])
            grid = np.array([[x, y, z0, s] for (x, y) in xy for s in speeds])
            ok = predict_feasible(net, to_observations(grid, rng)).reshape(len(dd), len(speeds)) >= .5
            limit = np.array([speeds[np.nonzero(row)[0].max()] if row.any() else np.nan for row in ok])
            ax.plot(dd, limit, color=color, alpha=.7, lw=1.5, label='learned limit (4 directions)' if angle == 0 else None)
    ax.plot(dd, rule.speed_limit(dd) * (dd >= rule.margin), 'k', lw=2, label='true limit')
    ax.axvline(rule.margin, color='k', ls='--', lw=1)
    ax.axhline(rule.v_cap, color='tab:blue', ls=':')
    ax.set(xlim=(0, 0.3), ylim=(0, 1.3 * rule.v_cap), xlabel='distance to cylinder border d (m)', ylabel='largest speed accepted (m/s)',
           title='position+speed network: accepted speed vs. distance')
    ax.legend(fontsize=8, loc='upper left')

    # (3) held-out errors in the (d, v) plane
    ax = axes[2]
    pts, truth = data['test_points'], data['test_label_position_speed']
    pred = predict_feasible(nets['position_speed'], to_observations(pts, rng)) >= .5
    d, v = rule.distance(pts[:, :2]), pts[:, 3]
    ax.scatter(d[truth == pred], v[truth == pred], s=1, color='lightgrey', label='correct')
    ax.scatter(d[~truth & pred], v[~truth & pred], s=10, color='tab:red', label='false feasible (%d)' % (~truth & pred).sum())
    ax.scatter(d[truth & ~pred], v[truth & ~pred], s=10, color='tab:orange', label='false infeasible (%d)' % (truth & ~pred).sum())
    dd = np.linspace(0, 0.3, 300)
    ax.plot(dd, rule.speed_limit(dd) * (dd >= rule.margin), 'k', lw=1.5)
    ax.set(xlim=(-0.04, 0.3), ylim=(0, v.max() * 1.02), xlabel='d (m)', ylabel='v (m/s)', title='held-out errors of the position+speed network')
    ax.legend(fontsize=8, loc='upper left')
    fig.savefig(os.path.join(out, 'learned.png'), bbox_inches='tight', dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- simulated planning
def resample_path(path, spacing):
    """Uniform arc-length resampling of a polyline (keeps both end points)."""
    path = np.asarray(path, float)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    n = max(int(np.ceil(s[-1] / spacing)), 1)
    grid = np.linspace(0.0, s[-1], n + 1)
    return np.column_stack([np.interp(grid, s, path[:, i]) for i in range(3)])


def accepted_speed(net, positions, speed_max, step=0.002):
    """Largest speed (m/s) the network accepts at each position: the last speed before it first says infeasible."""
    speeds = np.arange(0.0, speed_max + 1e-9, step)
    rng = np.random.RandomState(0)
    limits = np.zeros(len(positions))
    for i in range(0, len(positions), 400):
        chunk = positions[i:i + 400]
        grid = np.array([[*p, v] for p in chunk for v in speeds])
        ok = (predict_feasible(net, to_observations(grid, rng)) >= 0.5).reshape(len(chunk), len(speeds))
        first_bad = np.where(ok.all(1), len(speeds), (~ok).argmax(1))
        limits[i:i + 400] = np.where(first_bad == 0, 0.0, speeds[np.clip(first_bad - 1, 0, len(speeds) - 1)])
    return limits


def retime_along_path(path, limit, a_max):
    """Speed profile along a fixed geometric path: v_i <= limit_i, |dv/dt| <= a_max, rest at both ends.

    Forward/backward pass on v^2 (v_{i+1}^2 <= v_i^2 + 2 a ds). Returns the speed at each path point and time stamps.
    """
    path = np.asarray(path, float)
    ds = np.linalg.norm(np.diff(path, axis=0), axis=1)
    v2 = np.asarray(limit, float) ** 2
    v2[0] = v2[-1] = 0.0
    for i in range(len(ds)):
        v2[i + 1] = min(v2[i + 1], v2[i] + 2 * a_max * ds[i])
    for i in range(len(ds) - 1, -1, -1):
        v2[i] = min(v2[i], v2[i + 1] + 2 * a_max * ds[i])
    v = np.sqrt(v2)
    mean = np.maximum((v[:-1] + v[1:]) / 2, 1e-4)
    return v, np.r_[0.0, np.cumsum(ds / mean)]


def constant_speed_times(path, speed, ramp=0.03):
    """What the robot executes today: constant average speed (plus the short start/stop ramp of ``task_start_ramp_m``)."""
    ds = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(ds)]
    scale = np.minimum(1.0, np.minimum(s, s[-1] - s) / ramp).clip(0.2, 1.0)
    v = speed * (scale[:-1] + scale[1:]) / 2
    return np.r_[0.0, np.cumsum(ds / v)]


def check_timed(path, times, rule, net=None):
    """True-rule check of a timed polyline. Segment speed is compared to the true limit at the segment midpoint."""
    path, times = np.asarray(path, float), np.asarray(times, float)
    delta, dt = np.diff(path, axis=0), np.maximum(np.diff(times), 1e-9)
    speed = np.linalg.norm(delta, axis=1) / dt
    planar = np.linalg.norm(delta[:, :2], axis=1) / dt
    mid = (path[:-1] + path[1:]) / 2
    d = rule.distance(mid)
    limit = rule.speed_limit(d) * (d >= rule.margin)
    over = (planar > limit + 1e-3) | (d < rule.margin)
    over_5 = (planar > limit + 5e-3) | (d < rule.margin)      # beyond the boundary accuracy of a trained network
    accel = np.abs(np.diff(planar)) / ((dt[:-1] + dt[1:]) / 2)
    result = dict(duration_s=float(times[-1]), length_m=float(np.linalg.norm(delta, axis=1).sum()),
                  max_speed=float(speed.max()), max_planar_speed=float(planar.max()), max_accel=float(accel.max() if len(accel) else 0.0),
                  min_distance_to_border_m=float(d.min()), min_clearance_to_margin_m=float((d - rule.margin).min()),
                  violation_fraction=float(over.mean()), violation_fraction_gt_5mms=float(over_5.mean()), violation_length_m=float(np.linalg.norm(delta, axis=1)[over].sum()),
                  worst_speed_excess=float((planar - limit).max()))
    if net is not None:
        pts = np.column_stack([mid, planar])
        result['min_net_score'] = float(predict_feasible(net, to_observations(pts, np.random.RandomState(0))).min())
    return result, dict(speed=speed, planar=planar, d=d, limit=limit, mid=mid, over=over, t=(times[:-1] + times[1:]) / 2)


def robot_geometric_planner(position_pt, library, start_pose, goal_pose):
    """The robot repository's own offline planner on a position checkpoint (the checkpoint is copied into ``library``;
    nothing is written into the robot repository)."""
    sys.path.insert(0, os.path.join(ROBOT_ROOT, 'ros_ws/src/constraint_planner/src'))
    import shutil
    _load = th.load                           # the robot loader passes weights_only=, which torch < 1.13 does not know
    def compatible_load(f, *a, **k):
        start = f.tell() if hasattr(f, 'tell') else None
        try:
            return _load(f, *a, **k)
        except TypeError:
            k.pop('weights_only', None)
            if start is not None:
                f.seek(start)
            return _load(f, *a, **k)
    th.load = compatible_load
    from constraint_planner import configure_planning_profile, create_planner
    from constraint_planner.neural_constraint import NeuralSafetyConstraint
    os.makedirs(os.path.join(library, 'obsavoid'), exist_ok=True)
    source = os.path.join(library, 'obsavoid', os.path.basename(position_pt))
    shutil.copyfile(position_pt, source)
    task = json.load(open(ROBOT_TASK))
    configure_planning_profile(task, 'obsavoid', source, library)
    NeuralSafetyConstraint(task['neural_constraint'], library, 'iiwa7_link_0', 'obsavoid')   # the robot's own loader accepts the file
    keys = ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw')
    return create_planner(task), np.array([start_pose[k] for k in keys]), np.array([goal_pose[k] for k in keys])


SCENARIOS = {
    'A': 'A: robot today: position .pt, constant speed',
    'A2': 'A2: robot today: inflated position .pt, constant speed',
    'B': 'B: inflated geometry, retimed by the speed network',
    'C': 'C: joint path+timing optimiser (reference)',
}
COLORS = {'A': 'tab:red', 'A2': 'tab:orange', 'B': 'tab:blue', 'C': 'tab:green'}


def cmd_simulate(args):
    from icrl.nominal_planner import Trajectory_Optimizer
    out = os.path.abspath(args.out)
    config, rule, data, nets = load_run(out)
    box_low, box_high = data['box_low'], data['box_high']
    speed_net = nets['position_speed']
    task = json.load(open(os.path.join(EXPERT_PATH, 'task_config.json')))
    pairs = [(np.array(t['start']), np.array(t['goal']), 'demo %d' % i) for i, t in enumerate(task['tasks'][:args.n_demo_tasks])]
    poses = robot_poses()
    if poses:
        pairs.append((np.array([poses[0][k] for k in 'xyz']), np.array([poses[1][k] for k in 'xyz']), 'robot GUI task'))

    env = SimpleNamespace(step_size=rule.v_cap * DT, dt=DT, workspace_low=box_low, workspace_high=box_high, max_steps=args.max_steps)
    # The planner's a_max constraint does not converge reliably (see summary), so acceleration is imposed by retiming
    # and merely reported for the planner-timed reference C.
    planners = {}
    for key in ('position', 'position_inflated', 'position_speed'):
        planners[key] = Trajectory_Optimizer(env, v_max=rule.v_cap, duration_slack=(1.0, 1.0), bump=args.bump)
        planners[key].modulate_with_NN(nets[key].build_gamma_with_grad_for_ds)

    library = os.path.join(out, '_robot_library')
    results, curves = {}, {}
    for start, goal, name in pairs:
        print('==', name, np.round(start, 3), '->', np.round(goal, 3))
        row, paths = {}, {}
        for key in ('position', 'position_inflated'):
            if name == 'robot GUI task':
                planner, s7, g7 = robot_geometric_planner(config['files'][key]['path'], library, poses[0], poses[1])
                planned = planner.plan(s7, g7, None, None, seed=2026)
                raw = planned['positions'][:int(planned['stage_boundaries'][0]) + 1][:, :3]
                row['geometry_source_' + key] = 'robot repository planner'
            else:
                raw = planners[key].plan(start, goal).positions
                row['geometry_source_' + key] = 'pucl planner'
            paths[key] = resample_path(raw, args.spacing)
        # A: the robot as it runs today with the plain position checkpoint; A2: same with the inflated-border checkpoint
        row['A'], curves[(name, 'A')] = check_timed(paths['position'], constant_speed_times(paths['position'], args.task_speed), rule, speed_net)
        row['A2'], curves[(name, 'A2')] = check_timed(paths['position_inflated'], constant_speed_times(paths['position_inflated'], args.task_speed), rule, speed_net)
        # B: retime along the inflated geometry so that the speed stays under what the position+speed network accepts
        path = paths['position_inflated']
        limit = np.minimum(accepted_speed(speed_net, path, rule.v_cap), rule.v_cap) * args.safety
        _, t_retime = retime_along_path(path, limit, args.a_max)
        row['B'], curves[(name, 'B')] = check_timed(path, t_retime, rule, speed_net)
        # C: joint optimisation of path and timing under the position+speed network (time-optimal on the dt grid)
        plan = planners['position_speed'].plan(start, goal)
        row['C'], curves[(name, 'C')] = check_timed(plan.positions, np.arange(len(plan.positions)) * DT, rule, speed_net)
        row['C']['solver_feasible'] = bool(plan.feasible)
        curves[(name, 'A')]['path'], curves[(name, 'A2')]['path'] = paths['position'], paths['position_inflated']
        curves[(name, 'B')]['path'], curves[(name, 'C')]['path'] = path, plan.positions
        results[name] = row
        for key in SCENARIOS:
            r = row[key]
            print('  %-3s T=%5.1fs  vmax=%.3f  amax=%.2f  violating %5.1f%% (>5mm/s: %4.1f%%)  worst excess %+.3f  d_min %.3f' % (
                key, r['duration_s'], r['max_planar_speed'], r['max_accel'], 100 * r['violation_fraction'], 100 * r['violation_fraction_gt_5mms'], r['worst_speed_excess'],
                r['min_distance_to_border_m']))
    json.dump(dict(rule=rule.to_dict(), a_max=args.a_max, task_speed=args.task_speed, safety=args.safety, results=results),
              open(os.path.join(out, 'simulation.json'), 'w'), indent=2)
    plot_simulation(out, rule, pairs, curves, results, box_low, box_high)


def plot_simulation(out, rule, pairs, curves, results, box_low, box_high):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    names = [p[2] for p in pairs]
    show = [n for n in names if n in ('demo 0', 'demo 3', 'demo 8', 'robot GUI task')] or names[:3]
    fig, axes = plt.subplots(1, len(show), figsize=(5.2 * len(show), 5.6), squeeze=False)
    for ax, name in zip(axes[0], show):
        ax.add_patch(Circle(rule.center, rule.radius, color='0.8'))
        ax.add_patch(Circle(rule.center, rule.radius + rule.margin, fill=False, ls='--', ec='k'))
        for key in SCENARIOS:
            path = curves[(name, key)]['path']
            ax.plot(path[:, 0], path[:, 1], color=COLORS[key], lw=2 if key == 'B' else 1.2, label=SCENARIOS[key])
            bad, mid = curves[(name, key)]['over'], curves[(name, key)]['mid']
            ax.plot(mid[bad, 0], mid[bad, 1], 'x', color='k', ms=3)
        start = [p for p in pairs if p[2] == name][0]
        ax.plot(*start[0][:2], 'ko'); ax.plot(*start[1][:2], 'k*', ms=11)
        ax.set(aspect='equal', title=name + '  (x = violates the true rule)', xlabel='x (m)', ylabel='y (m)',
               xlim=(box_low[0], box_high[0]), ylim=(box_low[1], box_high[1]))
    axes[0][0].legend(fontsize=6.5, loc='lower left')
    fig.savefig(os.path.join(out, 'sim_paths.png'), bbox_inches='tight', dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(len(show), 3, figsize=(17, 3.9 * len(show)), squeeze=False)
    dd = np.linspace(0, 0.3, 300)
    for row, name in zip(axes, show):
        row[0].plot(dd, rule.speed_limit(dd) * (dd >= rule.margin), 'k', lw=2, label='true limit')
        for key in SCENARIOS:
            c = curves[(name, key)]
            row[0].plot(c['d'], c['planar'], '.', color=COLORS[key], ms=3, label=key)
            row[1].plot(c['t'], c['planar'], color=COLORS[key], label=key)
            row[2].plot(c['t'][1:], np.diff(c['planar']) / np.diff(c['t']), color=COLORS[key], lw=.8)
        row[0].set(xlim=(0, 0.3), ylim=(0, 0.22), xlabel='distance to cylinder border d (m)', ylabel='planar speed (m/s)', title=name + ': speed vs distance')
        row[0].legend(fontsize=8, loc='upper left')
        row[1].axhline(rule.v_cap, color='k', ls=':')
        row[1].set(xlabel='time (s)', ylabel='planar speed (m/s)', title='speed over time (dotted: v_cap)')
        row[2].set(xlabel='time (s)', ylabel='acceleration (m/s²)', title='acceleration (finite differences, noisy)', ylim=(-3, 3))
    fig.subplots_adjust(hspace=0.5)
    fig.savefig(os.path.join(out, 'sim_profiles.png'), bbox_inches='tight', dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    x, keys = np.arange(len(names)), list(SCENARIOS)
    width = 0.8 / len(keys)
    for i, key in enumerate(keys):
        offset = (i - (len(keys) - 1) / 2) * width
        axes[0].bar(x + offset, [100 * results[n][key]['violation_fraction'] for n in names], width, color=COLORS[key], label=SCENARIOS[key])
        axes[1].bar(x + offset, [results[n][key]['duration_s'] for n in names], width, color=COLORS[key])
        axes[2].bar(x + offset, [results[n][key]['max_planar_speed'] for n in names], width, color=COLORS[key])
    axes[0].set(ylabel='% of path violating the true rule', title='true-rule violations')
    axes[1].set(ylabel='s', title='duration')
    axes[2].set(ylabel='m/s', title='max planar speed (dotted: v_cap)')
    axes[2].axhline(rule.v_cap, color='k', ls=':')
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[0].legend(fontsize=7)
    fig.savefig(os.path.join(out, 'sim_summary.png'), bbox_inches='tight', dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train')
    train.add_argument('--out', default=os.path.join(ROOT, 'icrl/side_eval/run1'))
    train.add_argument('--expert_type', default='EXPERT')
    train.add_argument('--margin', type=float, default=0.02, help='added to the cylinder radius (m)')
    train.add_argument('--gain', type=float, default=1.5, help='speed limit slope (m/s per m) behind the margin border')
    train.add_argument('--v_cap', type=float, default=0.15, help='largest speed the robot may use (m/s)')
    train.add_argument('--speed_range', type=float, default=1.3, help='sample speeds up to this multiple of v_cap')
    train.add_argument('--select_dim', type=int, nargs='+', default=[0, 1, 2, 11], help='-cosd of the position+speed network')
    train.add_argument('--task_speed', type=float, default=0.084, help='constant speed the robot runs today (m/s); sets the inflated border')
    train.add_argument('--hidden', type=int, nargs='+', default=[64, 64, 64])
    train.add_argument('--n_uniform', type=int, default=40000)
    train.add_argument('--n_boundary', type=int, default=40000)
    train.add_argument('--pad_xy', type=float, default=0.06)
    train.add_argument('--pad_z', type=float, default=0.05)
    train.add_argument('--epochs', type=int, default=60)
    train.add_argument('--lr', type=float, default=3e-3)
    train.add_argument('--batch', type=int, default=2048)
    train.add_argument('--seed', type=int, default=0)
    simulate = sub.add_parser('simulate')
    simulate.add_argument('--out', default=os.path.join(ROOT, 'icrl/side_eval/run1'))
    simulate.add_argument('--n_demo_tasks', type=int, default=10)
    simulate.add_argument('--a_max', type=float, default=0.5, help='Cartesian acceleration limit assumed for the robot (m/s^2)')
    simulate.add_argument('--task_speed', type=float, default=0.084, help='robot task_speed_mps (constant speed used today)')
    simulate.add_argument('--safety', type=float, default=1.0, help='scale on the network-accepted speed when retiming')
    simulate.add_argument('--spacing', type=float, default=0.005, help='path resampling spacing (m)')
    simulate.add_argument('--max_steps', type=int, default=600)
    simulate.add_argument('--bump', type=float, default=0.16, help='sideways bump (m) of the planner initial guess; must clear the border of the inflated network')
    args = parser.parse_args()
    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'simulate':
        cmd_simulate(args)
