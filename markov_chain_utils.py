from typing import Tuple, Any

import numpy as np
from scipy.linalg import lstsq
from numpy import floating

from config import Config


def sigmoid(x: np.float32) -> np.float32:
    return 1 / (1 + np.exp(-x))


def identity(x: np.float32) -> np.float32:
    return x


def make_symmetric_transition_matrix(
    dimension: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate a symmetric transition matrix
    :param dimension:
    :param rng: numpy pseudo random number generator
    :return: P: dimension x dimension
    """
    random_matrix = rng.uniform(size=(dimension, dimension)).astype(np.float32)
    symmetric_transition_matrix = 0.5 * (random_matrix + random_matrix.T)

    return symmetric_transition_matrix / np.sum(symmetric_transition_matrix, axis=1)[:, None]


def make_2_state_transition_matrix(
    p: float
) -> np.ndarray:
    """
    Generate a 2-state transition matrix
    :param p: probability of going from state 1 to state 2
    :return: the transition matrix
    """
    return np.array([
        [1 - p, p],
        [p, 1 - p]
    ])


class MarkovChain:
    def __init__(
        self,
        gen_seed: int,
        config: Config
    ):
        self.rng = np.random.default_rng(gen_seed)

        # transition matrix of size (n_clients x n_states x n_states)
        self.transition_matrix = np.array([
            make_2_state_transition_matrix(1 / t)
            for t in config.mixing_times
        ])
        self.buffer_length = config.buffer_length
        self.stationary = config.stationary
        self.n_clients = config.n_clients
        self.data_dim = config.data_dim
        self.n_states = config.n_states
        self.regularization = config.regularization
        self.lambda_ = config.lambda_
        self.independent_batch = config.independent_batch
        if config.local_steps:
            self.batch_size = config.local_steps

        # Generating the data space of shape n_clients x n_states x data_dim
        common_data_space = self.rng.uniform(size=(config.n_states, config.data_dim))
        data_space_perturbation = self.rng.uniform(low=0.0, high=config.perturbed_scale,
                                                   size=(config.n_clients,
                                                         config.n_states, config.data_dim))

        # Perturbing and normalizing the data space with a uniform noise in [0, perturbed_scale)
        self.data_spaces = np.broadcast_to(common_data_space, (
            config.n_clients, config.n_states, config.data_dim)) + data_space_perturbation

        if config.normalization:
            self.data_spaces /= np.sum(self.data_spaces ** 2, axis=-1,
                                       keepdims=True)

        # Generating the optimal parameters of shape n_clients x n_states x data_dim
        if not config.heterogeneous:
            common_optimal_parameter = self.rng.uniform(size=(config.n_states, config.data_dim))
            optimal_parameter_perturbation = self.rng.uniform(low=0.0, high=config.perturbed_scale,
                                                              size=(config.n_clients, config.n_states, config.data_dim))

            # Perturbing and normalizing the optimal parameters with a uniform noise in [0, perturbed_scale]
            self.optimal_parameters = np.broadcast_to(common_optimal_parameter, (
                config.n_clients, config.n_states, config.data_dim
            )) + optimal_parameter_perturbation

            if config.normalization:
                self.optimal_parameters /= np.sum(self.optimal_parameters ** 2, axis=-1,
                                                  keepdims=True)

        else:
            optimal_parameter1 = self.rng.uniform(low=0., high=1., size=(config.n_states, config.data_dim))
            optimal_parameter2 = self.rng.uniform(low=1., high=2., size=(config.n_states, config.data_dim))

            optimal_parameter_perturbation = self.rng.uniform(low=0.0, high=config.perturbed_scale,
                                                              size=(config.n_clients, config.n_states, config.data_dim))
            self.optimal_parameters = np.zeros(shape=(config.n_clients, config.n_states, config.data_dim))

            for i in range(config.n_clients):
                self.optimal_parameters[i] = optimal_parameter1 if i % 2 == 0 else optimal_parameter2

            # Perturbing and normalizing
            self.optimal_parameters += optimal_parameter_perturbation

            if config.normalization:
                self.optimal_parameters /= np.sum(self.optimal_parameters ** 2, axis=-1, keepdims=True)

        # Generating the noises of shape n_clients x n_states x 1
        self.noises = self.rng.uniform(low=0., high=config.noise_scale,
                                       size=(config.n_clients, config.n_states, 1))

        # Generating the labels of shape n_clients x n_states x 1
        self.labels = np.sum(self.data_spaces * self.optimal_parameters, axis=-1, keepdims=True) + self.noises

        # Compute the minimum
        self.A = 1 / config.n_states * np.mean(
            np.matmul(np.transpose(self.data_spaces, [0, -1, 1]), self.data_spaces),
            axis=0
        )
        self.b = np.mean(
            self.labels * self.data_spaces,
            axis=(0, 1)
        )  # already divided by the number of states in np.mean
        try:
            self.minimum = np.linalg.solve(self.A, self.b)
        except np.linalg.LinAlgError:
            print("XX^T is singular")
            self.minimum = lstsq(self.A, self.b)[0]

        # Get the latest state of the chain
        self.latest_state = None

    def sample(
        self
    ) -> Tuple[np.ndarray, np.ndarray]:

        """
        Sample trajectories from the Markov chain of all clients
        :return: data_stream of shape (n_clients, buffer_length, data_dim)
                 label of shape (n_clients, buffer_length, 1) # need to check on this
        """

        data_buffer, label_buffer = (np.zeros((self.n_clients, self.buffer_length, self.data_dim)),
                                     np.zeros((self.n_clients, self.buffer_length, 1)))

        # If we are sampling for the first time, set the beginning state based on
        # whether the chain is stationary or not

        for step in range(self.buffer_length):
            if self.latest_state is None:
                if self.stationary:
                    self.latest_state = np.array([
                        self.rng.choice(self.n_states)
                        for _ in range(self.n_clients)
                    ])
                else:
                    self.latest_state = np.zeros((self.n_clients,)).astype(np.int32)

            else:
                transition_probs = self.transition_matrix[np.arange(self.n_clients), self.latest_state]  # (n_clients x n_states)
                next_state = ((self.rng.uniform(size=(self.n_clients, 1)) < transition_probs.cumsum(axis=-1)).
                              argmax(axis=-1))
                self.latest_state = next_state

            data_buffer[:, step] = self.data_spaces[np.arange(self.n_clients), self.latest_state]
            label_buffer[:, step] = self.labels[np.arange(self.n_clients), self.latest_state]

            if self.independent_batch and (step + 1) % self.batch_size == 0:
                self.reset()

        return data_buffer, label_buffer

    def evaluate(
        self,
        w: np.ndarray
    ) -> tuple[float, floating[Any]]:
        """
        Compute the gradient norm of the global objective function w.r.t the model w
        :param w: the global model of shape (data_dim)
        :return: the gradient norm || nabla F(w) ||^2
        """
        loss = 0.5 * np.mean(
            (np.dot(self.data_spaces, w)[..., None] - self.labels) ** 2
        )  # already divided by the number of state in np.mean

        grad = np.dot(self.A, w) - self.b

        if self.regularization:
            loss += 0.5 * self.lambda_ * np.sum(w**2/(1+w**2))
            grad += self.lambda_ * (w/(1+w**2)**2)

        grad_norm = np.linalg.norm(grad) ** 2

        return loss, grad_norm

    def set_rng(
        self,
        seed: int
    ):
        self.rng = np.random.default_rng(seed)

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size

    def reset(self):
        self.latest_state = None
