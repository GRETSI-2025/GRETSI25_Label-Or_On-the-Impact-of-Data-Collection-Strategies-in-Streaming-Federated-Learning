import os

import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl

import pickle

import seaborn as sns

mpl.rcParams.update({'font.size': 12})
palette = sns.color_palette("colorblind")

stream_length = 500000
global_lr = 0.01
local_lr = 1e-3
momentum_coef = 0.1
h = "heterogeneous"

metric = "grad_norm"
algos = ["local_sgd", "minibatch_sgd"]
colors_independent_batch = {
    "local_sgd": {
        True: palette[0],
        False: palette[1],
    },
    "minibatch_sgd": {
        True: palette[2],
        False: palette[3],
    }
}
markers_independent_batch = {
    "local_sgd": {
        True: "o",
        False: "D",
    },
    "minibatch_sgd": {
        True: "s",
        False: "v",
    }
}
label_independent_batch = {
    "local_sgd": {
        True: "Local SGD Independent Batch",
        False: "Local SGD Dependent Batch",
    },
    "minibatch_sgd": {
        True: "Minibatch SGD Independent Batch",
        False: "Minibatch SGD Dependent Batch",
    }
}


def ema(scalars, weight=0.5):
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return np.array(smoothed)


def smooth(scalars, weight=0.1, start=0):
    return np.concatenate((scalars[:start], ema(scalars[start:], weight)))


def smooth_upper(scalars, variance, weight=0.1, start=0):
    return smooth(scalars + variance, weight=weight, start=start)


def smooth_lower(scalars, variance, min_val, weight=0.1, start=0):
    return smooth(np.maximum(scalars - variance, min_val), weight=weight, start=start)


def plot_independent_batch(
    dependent_res_dir: str,
    independent_res_dir: str
):
    plot_dir = "plot/independent_batch"

    independent_batch_local_steps_list = [10, 100, 1000]
    independent_batch_n_clients_list = [100]
    independent_batch_mixing_time = [10, 100, 1000, 10000]

    for local_steps in independent_batch_local_steps_list:
        for n_clients in independent_batch_n_clients_list:
            for mixing_time in independent_batch_mixing_time:
                for algo in algos:
                    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
                    ax.set_yscale("log", base=10)
                    plt.xlabel("Communication rounds")
                    for independent_batch in [True, False]:
                        f_name = f"mixing_time={mixing_time},local_lr={local_lr},global_lr={global_lr},momentum={momentum_coef},local_steps={local_steps},n_communications={stream_length // local_steps},n_clients={n_clients}.pkl"

                        if independent_batch:
                            with open(os.path.join(independent_res_dir, f_name), "rb") as f:
                                results = pickle.load(f)
                        else:
                            with open(os.path.join(dependent_res_dir, f_name), "rb") as f:
                                results = pickle.load(f)

                        grad_norm = results[algo]["grad_norm"]
                        grad_norm_mean = np.mean(grad_norm, axis=0)
                        grad_norm_std = np.std(grad_norm, axis=0)
                        grad_norm_min = np.min(grad_norm, axis=0)

                        plot_idx = np.linspace(0, len(grad_norm_mean) - 1,
                                               min(len(grad_norm_mean), 200), dtype=int)

                        plt.plot(plot_idx, smooth(grad_norm_mean[plot_idx], weight=0.1),
                                 color=colors_independent_batch[algo][independent_batch],
                                 marker=markers_independent_batch[algo][independent_batch],
                                 label=label_independent_batch[algo][independent_batch],
                                 markevery=len(plot_idx) // 10)

                        plt.fill_between(plot_idx,
                                         smooth_lower(grad_norm_mean[plot_idx], grad_norm_std[plot_idx],
                                                      grad_norm_min[plot_idx], weight=0.1),
                                         smooth_upper(grad_norm_mean[plot_idx], grad_norm_std[plot_idx], weight=0.1),
                                         color=colors_independent_batch[algo][independent_batch],
                                         edgecolor=colors_independent_batch[algo][independent_batch], alpha=0.5)

                    plot_subdir = f"{plot_dir}/mixing_time={mixing_time}/{algo}"
                    os.makedirs(plot_subdir, exist_ok=True)
                    plt.savefig(f"{plot_subdir}/M={n_clients},K={local_steps}.pdf", bbox_inches='tight')
                    plt.close()
    # plot the legend
    handles = []
    labels = []
    for algo in algos:
        for independent_batch in [True, False]:
            handles.append(plt.plot([], [], marker=markers_independent_batch[algo][independent_batch],
                                    color=colors_independent_batch[algo][independent_batch])[0])
            labels.append(label_independent_batch[algo][independent_batch])

    legend = plt.legend(handles, labels, frameon=False, ncol=4)
    plt.axis("off")
    fig = legend.figure
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("plot/independent_batch_legend.pdf", dpi="figure", bbox_inches=bbox)


if __name__ == "__main__":
    dependent_res_dir = "results/dependent_batch"
    independent_res_dir = "results/independent_batch"
    plot_independent_batch(
        dependent_res_dir,
        independent_res_dir
    )
