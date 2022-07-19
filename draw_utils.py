# code from "https://github.com/guanyingc/UPS-GCNet/blob/master/utils/draw_utils.py"
import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import os

def set_figscale(fig, ax):
    x0, y0, dx, dy = ax.get_position().bounds
    w = 3 * max(dx, dy) /dx
    h = 3 * max(dx, dy) /dy
    fig.set_size_inches((w, h))

def draw_circle(ax):
    t = np.linspace(0, 2 * np.pi, 200)
    x, y = np.cos(t), np.sin(t)
    ax.plot(x*1.0, y*1.0, 'k')
    axis = 1.01
    ax.axis([-axis, axis, -axis, axis])

def plot_light(x, y, save_name, c=None):
    # If the input is from MLP network, need to convert y = -y
    y = -y
    fig, ax = plt.subplots()

    if c is None:
        ax.scatter(x, y, s=6)
    else:
        plt.scatter(x, y, c=c, cmap='jet', vmin=0, vmax=1)
    draw_circle(ax)

    ax.axis('off')
    set_figscale(fig, ax)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(save_name, bbox_inches=extent)
    plt.close()

def plot_lighting(dirs, ints, save_dir):
    # Visualize light direction and intensity
    save_name = os.path.join(save_dir, 'est_light_map.png')
    if len(ints.shape) > 1:
        ints = ints.mean(-1)
    ints = ints / ints.max()
    plot_light(dirs[:,0], dirs[:, 1], save_name, ints)

def plot_lighting_gt(dirs, ints, save_dir):
    # Visualize light direction and intensity
    save_name = os.path.join(save_dir, 'est_light_map_gt.png')
    if len(ints.shape) > 1:
        ints = ints.mean(-1)
    ints = ints / ints.max()
    plot_light(dirs[:,0], dirs[:, 1], save_name, ints)

def plot_dir_error(light, error, save_dir):
    # plot light direction estimation error
    save_name = os.path.join(save_dir, 'est_light_error_dir.png')
    error = error / 25
    plot_light(light[:,0], light[:, 1], save_name, error)

def plot_int_error(light, error, save_dir):
    # plot light intensity estimation error
    save_name = os.path.join(save_dir, 'est_light_error_int.png')
    error = error / 0.2
    plot_light(light[:,0], light[:, 1], save_name, error)
