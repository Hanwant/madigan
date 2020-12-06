from typing import Union

import numpy as np
import torch


def discrete_action_to_transaction(
        actions: torch.Tensor, action_atoms: int, margin_prop: float,
        avail_margin: float, prices: Union[np.ndarray,
                                           torch.Tensor]) -> np.ndarray:
    """
    Given integer actions (I.e class indexes of NN output)
    Computes corresponding transaction amounts
    Assumes an odd number of action_atoms to allow for hold (0 transaction)

    transaction amounts are computed by taking a proportion of available margin
    I.e
        transaction_value = margin_prop * avail_margin
        transaction_units = transaction_value / prices

    """
    if action_atoms % 2 == 0:
        raise ValueError("action atoms should be an odd number")
    units = margin_prop * avail_margin / prices
    actions_centered = (actions - (action_atoms // 2))
    if isinstance(actions_centered, torch.Tensor):
        actions_centered = actions_centered.cpu().numpy()
    return actions_centered * units


def abs_port_norm(port: torch.Tensor) -> torch.Tensor:
    """
    Given a portfolio Tensor, normalize with respect to sum(abs(port))
    This prevents numerical instability issues when using the normal port norm
    Where the portfolio is normed by the equity (I.e weights sum to 1.)

    Given that this normalization is done for NNs (not accounting), the
    input case is expected to be a batched tensor and so the norm is done
    with respect to the last dim
    """
    return port / port.abs().sum(-1, keepdim=True)


def abs_port_denorm(abs_norm_port: torch.Tensor) -> torch.Tensor:
    """
    Performs inverse operation to abs_norm_port
    To recover original portfolio
    """
    return abs_norm_port * 1 / abs_norm_port.sum(-1, keepdim=True)
