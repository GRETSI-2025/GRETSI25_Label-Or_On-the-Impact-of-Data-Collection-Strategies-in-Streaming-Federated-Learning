import functools
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

import numpy as np

from algos import local_sgd, minibatch_sgd, local_sgd_momentum
from config import Config
from markov_chain_utils import MarkovChain

repeat = 5
seed_value = 0


def seed():
    global seed_value
    seed_value += 1
    return seed_value


def run_algo(
    s: int,
    algo,
    config: Config,
    common_chain: MarkovChain,
    w0: np.ndarray
):
    markov_chain = deepcopy(common_chain)
    markov_chain.set_rng(s)
    markov_chain.reset()

    # print(f"Running {algo.__name__} with seed = {s}")

    return (
        algo.__name__,
        algo(config, markov_chain, w0)
    )


def seed_map(
    t
):
    """
    Apply the function t[0] with the seed t[1]
    :param t: Tuple of [callable function, seed]
    :return:
    """
    return t[0](t[1])


def run_experiment(
    n_clients: int,
    mixing_time: int,
    local_steps: int,
    independent_batch: bool
):

    # global_lr_list = [0.01]
    # local_lr_list = [0.001]
    # momentum_coef_list = [0.1]
    global_lr, local_lr, momentum_coef = 0.01, 0.001, 0.1

    config = Config(
        global_lr=global_lr,
        local_lr=local_lr,
        momentum_coef=momentum_coef,
        noise_scale=1e-3,
        perturbed_scale=1e-2,
        heterogeneous=True,
        stationary=False,
        n_clients=n_clients,
        mixing_times=[mixing_time for _ in range(n_clients)],
        local_steps=local_steps,
        stream_length=500000,
        independent_batch=independent_batch
    )

    if independent_batch:
        config.stationary = True
        results_dir = "results/independent_batch"
    else:
        results_dir = "results/dependent_batch"

    common_markov_chain = MarkovChain(
        gen_seed=42,
        config=config
    )
    w0 = common_markov_chain.minimum

    os.makedirs(results_dir, exist_ok=True)

    n_communications = config.stream_length // config.local_steps

    if independent_batch:
        common_markov_chain.set_batch_size(local_steps)

    print(
        f"---- M = {n_clients}, mixing_time = {mixing_time}, T = {n_communications}, K = {config.local_steps}, local_lr = {config.local_lr}, global_lr = {config.global_lr}, momentum = {config.momentum_coef}, independent_batch = {config.independent_batch},----")

    func_local_sgd = functools.partial(
        run_algo,
        algo=local_sgd,
        config=config,
        common_chain=common_markov_chain,
        w0=w0 + 0.1
    )

    func_minibatch_sgd = functools.partial(
        run_algo,
        algo=minibatch_sgd,
        config=config,
        common_chain=common_markov_chain,
        w0=w0 + 0.1
    )

    func_momentum_sgd = functools.partial(
        run_algo,
        algo=local_sgd_momentum,
        config=config,
        common_chain=common_markov_chain,
        w0=w0 + 0.1
    )

    with ProcessPoolExecutor(max_workers=10) as executor:
        res = list(executor.map(
            seed_map,
            sum([
                [(func_local_sgd, seed()),
                 (func_minibatch_sgd, seed()),
                 (func_momentum_sgd, seed())]
                for _ in range(repeat)
            ], [])
        ))

    # markov_chain.reset()

    results_local_sgd = [r[1] for r in res if r[0] == "local_sgd"]
    results_minibatch_sgd = [r[1] for r in res if r[0] == "minibatch_sgd"]
    results_local_sgd_momentum = [r[1] for r in res if r[0] == "local_sgd_momentum"]

    w_norm_local_sgd, loss_local_sgd, grad_norm_local_sgd = np.zeros((repeat, n_communications+1)), np.zeros((repeat, n_communications+1)), np.zeros((repeat, n_communications+1))
    w_norm_minibatch_sgd, loss_minibatch_sgd, grad_norm_minibatch_sgd = np.zeros((repeat, n_communications+1)), np.zeros((repeat, n_communications+1)), np.zeros((repeat, n_communications+1))
    w_norm_local_sgd_momentum, loss_local_sgd_momentum, grad_norm_local_sgd_momentum = np.zeros((repeat, n_communications+1)), np.zeros((repeat, n_communications+1)), np.zeros((repeat, n_communications+1))

    for i in range(repeat):
        w_norm_local_sgd[i] = results_local_sgd[i][0]
        loss_local_sgd[i] = results_local_sgd[i][1]
        grad_norm_local_sgd[i] = results_local_sgd[i][2]
        # w_hist_local_sgd[i] = results_local_sgd[i][3]

        w_norm_minibatch_sgd[i] = results_minibatch_sgd[i][0]
        loss_minibatch_sgd[i] = results_minibatch_sgd[i][1]
        grad_norm_minibatch_sgd[i] = results_minibatch_sgd[i][2]

        w_norm_local_sgd_momentum[i] = results_local_sgd_momentum[i][0]
        loss_local_sgd_momentum[i] = results_local_sgd_momentum[i][1]
        grad_norm_local_sgd_momentum[i] = results_local_sgd_momentum[i][2]

    file_name = f"mixing_time={mixing_time},local_lr={config.local_lr},global_lr={config.global_lr},momentum={config.momentum_coef},local_steps={config.local_steps},n_communications={config.stream_length // config.local_steps},n_clients={n_clients}"

    with open(f"{results_dir}/{file_name}.pkl", "wb") as f:
        pickle.dump({
            "config": config,
            "local_sgd": {
                "w_norm": w_norm_local_sgd,
                "loss": loss_local_sgd,
                "grad_norm": grad_norm_local_sgd,
            },
            "minibatch_sgd": {
                "w_norm": w_norm_minibatch_sgd,
                "loss": loss_minibatch_sgd,
                "grad_norm": grad_norm_minibatch_sgd,
            },
            "local_sgd_momentum": {
                "w_norm": w_norm_local_sgd_momentum,
                "loss": loss_local_sgd_momentum,
                "grad_norm": grad_norm_local_sgd_momentum,
            }
        }, f)


if __name__ == "__main__":
    for t in [10, 100, 1000, 10000]:
        for local_steps in [10, 100, 1000]:
            for independent in [True, False]:
                run_experiment(100, t, local_steps, independent)
