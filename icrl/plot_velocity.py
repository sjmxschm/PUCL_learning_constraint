"""Plot the learned velocity constraint of KukaCup1 against distance to the obstacle and speed."""
import gym
import matplotlib
import numpy as np


def learned_speed_limit(cost_function, threshold, observations, acs_dim, dt, speed_max, n_speeds=60, chunk=20000):
    """Largest planar speed the network accepts at each observation's own position and heading.

    Each observation keeps its position and the direction of its own displacement (direction to the goal
    if it is standing still). Only the speed is swept from 0 to ``speed_max``. The limit is the last speed
    before the cost first exceeds ``threshold``: 0 if even a standstill is infeasible, ``speed_max`` if no speed is.
    """
    observations = np.asarray(observations, dtype=float)
    heading = observations[:, 6:8].copy()
    norm = np.linalg.norm(heading, axis=1, keepdims=True)
    fallback = observations[:, 3:5] / np.maximum(np.linalg.norm(observations[:, 3:5], axis=1, keepdims=True), 1e-9)
    heading = np.where(norm > 1e-9, heading / np.maximum(norm, 1e-12), fallback)
    speeds = np.linspace(0.0, speed_max, n_speeds)
    swept = np.repeat(observations, n_speeds, axis=0)
    swept[:, 6:8] = (heading[:, None, :] * speeds[None, :, None] * dt).reshape(-1, 2)
    swept[:, 8] = 0.0
    cost = np.concatenate([np.asarray(cost_function(swept[i:i + chunk], np.zeros((len(swept[i:i + chunk]), acs_dim)))).reshape(-1)
                           for i in range(0, len(swept), chunk)]).reshape(len(observations), n_speeds)
    infeasible = cost > threshold
    first = np.where(infeasible.any(axis=1), infeasible.argmax(axis=1), n_speeds)   # index of first infeasible speed
    return speeds[np.clip(first - 1, 0, n_speeds - 1)] * (first > 0)


def plot_velocity_constraint(cost_function, threshold, obs_dim, acs_dim, save_name, title,
                             obs_expert=None, obs_nominal=None, obs_rn=None, show=False):
    """Two panels, both with distance to the cylinder border d on the x axis and planar speed v on the y axis.

    LEFT, "what speed does the network allow?": for every expert point the network's maximum accepted speed at
    that point's own position and heading (grey dots; blue line = median per distance bin, band = 25-75%).
    A good constraint has this limit above the green expert dots (experts are feasible) and below the red
    reliable-infeasible dots, and it should rise with distance like the true limit (black line).

    RIGHT, "is each point judged correctly?": every point coloured by the network's own cost at its real
    observation (green feasible, red infeasible; circled = disagrees with its training label). Points above the
    black true-limit line should be red, points below it green. The two counts in the box measure that.

    d = |p_xy - centre| - radius, v = |obs[6:8]| / dt, true limit v = velocity_gain * d.
    """
    matplotlib.use('TkAgg' if show else 'Agg')
    import matplotlib.pyplot as plt
    env = gym.make('KukaCup1-v0')
    try:
        center, radius = np.asarray(env.obstacle_center), float(env.obstacle_radius)
        dt, gain, step = float(env.dt), float(env.velocity_gain), float(env.step_size)
    finally:
        env.close()
    speed_max = step * np.sqrt(2) / dt            # largest planar speed one action can produce

    groups = []   # label, colour, marker, size, observations, wrongly classified if cost above (True) / below (False) / never (None)
    for values, label, color, marker, size, wrong_if_infeasible in [
            (obs_expert, 'expert', 'green', 'o', 6, True), (obs_nominal, 'policy', 'orange', '.', 6, None),
            (obs_rn, 'reliable infeasible', 'red', '^', 10, False)]:
        if values is None or not np.asarray(values).size:
            continue
        values = np.asarray(values, dtype=float)
        point_cost = np.asarray(cost_function(values, np.zeros((len(values), acs_dim)))).reshape(-1)
        wrong = np.zeros(len(values), bool) if wrong_if_infeasible is None else (
            point_cost > threshold if wrong_if_infeasible else point_cost <= threshold)
        groups.append(dict(
            label=label if wrong_if_infeasible is None else '%s (%d of %d disagree with label)' % (label, wrong.sum(), len(wrong)),
            color=color, marker=marker, size=size, wrong=wrong, cost=point_cost, values=values,
            dist=np.linalg.norm(values[:, :2] - center, axis=1) - radius,
            speed=np.linalg.norm(values[:, 6:8], axis=1) / dt))

    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    d_line = np.linspace(0, 0.2, 100)
    for ax in (left, right):
        if np.isfinite(gain):
            ax.plot(d_line, np.minimum(gain * d_line, speed_max), color='black', lw=2, label='true limit v = %g * d' % gain)
        ax.set(xlim=(0, 0.2), ylim=(0, speed_max), xlabel='distance to cylinder border d (m)')

    # LEFT: learned limit at the real expert positions
    if groups and groups[0]['label'].startswith('expert'):
        expert = groups[0]
        limit = learned_speed_limit(cost_function, threshold, expert['values'], acs_dim, dt, speed_max)
        left.scatter(expert['dist'], limit, s=4, color='lightgrey', label='network limit at each expert point')
        edges = np.arange(0.05, 0.21, 0.01)
        centres, med, low, high = [], [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (expert['dist'] >= lo) & (expert['dist'] < hi)
            if m.sum() >= 8:
                centres.append((lo + hi) / 2); med.append(np.median(limit[m]))
                low.append(np.percentile(limit[m], 25)); high.append(np.percentile(limit[m], 75))
        if centres:
            left.fill_between(centres, low, high, color='tab:blue', alpha=0.25)
            left.plot(centres, med, color='tab:blue', lw=2.5, label='learned limit (median, 25-75%)')
    for g in groups:
        left.scatter(g['dist'], g['speed'], s=g['size'], color=g['color'], marker=g['marker'], label=g['label'])
    left.set_ylabel('planar speed v (m/s)')
    left.set_title('what speed does the network allow?')
    left.legend(loc='upper left', fontsize=8)

    # RIGHT: actual cost of every point, and agreement with the true limit
    image = None
    for g in groups:
        image = right.scatter(g['dist'], g['speed'], s=10 * g['size'], c=g['cost'], vmin=0, vmax=1, cmap='RdYlGn_r',
                              marker=g['marker'], edgecolors='none')
        right.scatter(g['dist'][g['wrong']], g['speed'][g['wrong']], s=45, facecolors='none', edgecolors='black', linewidths=0.8)
    others = [g for g in groups if not g['label'].startswith('expert')]
    if others and np.isfinite(gain):          # policy and reliable-infeasible points versus the true rule
        dist = np.concatenate([g['dist'] for g in others]); speed = np.concatenate([g['speed'] for g in others])
        judged_infeasible = np.concatenate([g['cost'] for g in others]) > threshold
        above = speed > gain * dist
        text = ('policy + reliable-infeasible points:\n'
                ' above true limit, judged feasible: %d of %d (missed violations)\n'
                ' below true limit, judged infeasible: %d of %d (over-cautious)'
                % ((above & ~judged_infeasible).sum(), above.sum(), (~above & judged_infeasible).sum(), (~above).sum()))
        experts = [g for g in groups if g['label'].startswith('expert')]
        if experts:
            text += '\nexperts above the true limit: %d of %d' % ((experts[0]['speed'] > gain * experts[0]['dist']).sum(),
                                                               len(experts[0]['speed']))
        right.text(0.98, 0.97, text, transform=right.transAxes, ha='right', va='top', fontsize=8,
                   bbox=dict(facecolor='white', alpha=0.85, edgecolor='grey'))
    right.set_title('is each point judged correctly?')
    if image is not None:
        fig.colorbar(image, ax=right, label='network cost at the point (threshold %.2f); circled = disagrees with label' % threshold)
    fig.suptitle(title)
    fig.savefig(save_name, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
