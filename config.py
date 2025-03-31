from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Config:
    global_lr: Optional[float]
    local_lr: Optional[float]
    momentum_coef: Optional[float]
    heterogeneous: Optional[bool]
    noise_scale: float
    perturbed_scale: float
    stationary: bool
    n_clients: Optional[int]
    mixing_times: List[int]
    local_steps: Optional[int]
    stream_length: Optional[int]
    regularization: bool = True
    lambda_: float = 0.01
    buffer_length: int = 1000
    n_states: int = 2
    data_dim: int = 10
    normalization: bool = False
    independent_batch: bool = False
