import numpy as np

from tqdm import tqdm

from config import Config
from markov_chain_utils import MarkovChain


def grad(
    X: np.ndarray,
    y: np.ndarray,
    ws: np.ndarray,
    regularization: bool,
    lambda_: float
) -> np.ndarray:
    """
    Compute the gradient of the loss function for all w
    :param lambda_: regularization coef
    :param regularization:
    :param X: the data of shape (n_clients, data_dim)
    :param y: the label of shape (n_clients, 1)
    :param ws: the parameters of shape (n_clients, data_dim)
    :return: the gradient of shape (n_clients, data_dim)
    """
    if not regularization:
        return (
            np.sum(ws * X, axis=-1, keepdims=True) - y
        ) * X
    else:
        return (
            np.sum(ws * X, axis=-1, keepdims=True) - y
        ) * X + lambda_ * np.sum(ws/(1+ws**2)**2, axis=-1, keepdims=True)


def local_sgd(
    config: Config,
    markov_chain: MarkovChain,
    w_0: np.ndarray
):
    n_communications = config.stream_length // config.local_steps
    buffer_count = 0

    w = w_0.copy()
    w_norm = np.zeros(n_communications + 1)
    w_norm[0] = np.linalg.norm(w - markov_chain.minimum)**2

    w_hist = np.zeros((n_communications + 1, config.data_dim))
    w_hist[0] = w

    loss = np.zeros(n_communications + 1)
    grad_norm = np.zeros(n_communications + 1)
    loss[0], grad_norm[0] = markov_chain.evaluate(w)

    for t in tqdm(range(n_communications)):
        ws = np.array([w.copy() for _ in range(config.n_clients)])

        for k in range(config.local_steps):
            if buffer_count == 0:
                data_buffer, label_buffer = markov_chain.sample()

            ws -= config.local_lr * grad(data_buffer[:, buffer_count, :], label_buffer[:, buffer_count, :], ws, config.regularization, config.lambda_)

            # update the buffer count
            buffer_count = (buffer_count + 1) % config.buffer_length

        # communicate
        w = np.mean(ws, axis=0)
        w_norm[t + 1] = np.linalg.norm(w - markov_chain.minimum)**2
        loss[t + 1], grad_norm[t + 1] = markov_chain.evaluate(w)
        w_hist[t + 1] = w

        # restart the sampling in the case of independence batch
        # if config.independent_batch:
        #     markov_chain.reset()

    return w_norm, loss, grad_norm, w_hist


def minibatch_sgd(
    config: Config,
    markov_chain: MarkovChain,
    w_0: np.ndarray
):
    n_communications = config.stream_length // config.local_steps
    buffer_count = 0

    w = w_0.copy()
    w_norm = np.zeros(n_communications + 1)
    w_norm[0] = np.linalg.norm(w - markov_chain.minimum)**2

    w_hist = np.zeros((n_communications + 1, config.data_dim))
    w_hist[0] = w

    loss = np.zeros(n_communications + 1)
    grad_norm = np.zeros(n_communications + 1)
    loss[0], grad_norm[0] = markov_chain.evaluate(w)

    for t in tqdm(range(n_communications)):
        client_grad = np.zeros((config.n_clients, config.data_dim))

        for k in range(config.local_steps):
            if buffer_count == 0:
                data_buffer, label_buffer = markov_chain.sample()

            client_grad += grad(data_buffer[:, buffer_count, :], label_buffer[:, buffer_count, :], w, config.regularization, config.lambda_)

            # Update the buffer count
            buffer_count = (buffer_count + 1) % config.buffer_length

        w -= config.global_lr * np.mean(client_grad, axis=0) / config.local_steps
        w_norm[t + 1] = np.linalg.norm(w - markov_chain.minimum)**2
        loss[t + 1], grad_norm[t + 1] = markov_chain.evaluate(w)
        w_hist[t + 1] = w

        # restart the sampling in the case of independence batch
        # if config.independent_batch:
        #     markov_chain.reset()

    return w_norm, loss, grad_norm, w_hist


def local_sgd_momentum(
    config: Config,
    markov_chain: MarkovChain,
    w_0: np.ndarray
):
    n_communications = config.stream_length // config.local_steps
    buffer_count = 0

    w = w_0.copy()
    w_norm = np.zeros(n_communications+1)
    w_norm[0] = np.linalg.norm(w - markov_chain.minimum)**2

    w_hist = np.zeros((n_communications + 1, config.data_dim))
    w_hist[0] = w

    # initialize the momentum
    momentum = np.zeros(config.data_dim)
    # initialize the global update
    global_update = np.zeros((config.n_clients, config.data_dim))

    loss = np.zeros(n_communications + 1)
    grad_norm = np.zeros(n_communications + 1)
    loss[0], grad_norm[0] = markov_chain.evaluate(w)

    for t in range(n_communications):
        # broadcast the updated global model
        ws = np.array([w.copy() for _ in range(config.n_clients)])

        for k in range(config.local_steps):
            if buffer_count == 0:
                data_buffer, label_buffer = markov_chain.sample()

            # compute the grad
            client_update = config.momentum_coef * grad(
                data_buffer[:, buffer_count, :],
                label_buffer[:, buffer_count, :],
                ws,
                config.regularization,
                config.lambda_
            ) + (1 - config.momentum_coef) * momentum

            # local update
            ws -= config.local_lr * client_update

            # store the local update
            global_update += client_update

            # update the buffer count
            buffer_count = (buffer_count + 1) % config.buffer_length

        # global update
        momentum = np.mean(global_update, axis=0) / config.local_steps
        w -= config.global_lr * momentum

        # reset the global update to zero
        global_update = np.zeros((config.n_clients, config.data_dim))

        # log
        w_norm[t+1] = np.linalg.norm(w - markov_chain.minimum)**2
        loss[t+1], grad_norm[t+1] = markov_chain.evaluate(w)
        w_hist[t + 1] = w

        # restart the sampling process in the case of independence batch
        # if config.independent_batch:
        #     markov_chain.reset()

    return w_norm, loss, grad_norm, w_hist
