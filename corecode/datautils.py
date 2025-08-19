"""
Utility functions for EA-LSTM
-----------------------------------------
This module provides helper functions for data scaling, attribute loading,
and evaluation metrics.

Functions
---------
• rescale_features : Rescale normalized features (e.g., GPP) back to natural units.
• load_attributes  : Load static attributes.
• calc_nse         : Compute Nash-Sutcliffe Efficiency between observations and simulations.
• calc_rmse        : Compute Root Mean Square Error.
• calc_mae         : Compute Mean Absolute Error.

Notes
-----
All metrics expect obs and sim as 1-D arrays of the same length.
"""


import numpy as np
import pandas as pd

SCALER = {'gpp_max': np.load('gpp_max.npy')}


def rescale_features(feature: np.ndarray, variable: str) -> np.ndarray:
    if variable == 'gpp':
        feature = feature * SCALER["gpp_max"]
    else:
        raise RuntimeError(f"Unknown variable type {variable}")
    return feature


def load_attributes(db_path: str) -> pd.DataFrame:
    df = pd.read_csv(db_path, index_col='SITE_ID')

    return df


def calc_nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Nash-Sutcliffe-Efficiency

    Parameters
    ----------
    obs : np.ndarray
        Array containing the discharge observations
    sim : np.ndarray
        Array containing the discharge simulations

    Returns
    -------
    float
        Nash-Sutcliffe-Efficiency

    Raises
    ------
    RuntimeError
        If `obs` and `sim` don't have the same length
    RuntimeError
        If all values in the observations are equal
    """
    # make sure that metric is calculated over the same dimension
    obs = obs.flatten()
    sim = sim.flatten()

    if obs.shape != sim.shape:
        raise RuntimeError("obs and sim must be of the same length.")

    # denominator of the fraction term
    denominator = np.sum((obs - np.mean(obs)) ** 2)

    # this would lead to a division by zero error and nse is defined as -inf
    if denominator == 0:
        return -np.inf

    # numerator of the fraction term
    numerator = np.sum((sim - obs) ** 2)

    # calculate the NSE
    nse_val = 1 - numerator / denominator

    return nse_val


def calc_rmse(obs: np.ndarray, sim: np.ndarray):
    """Calculate Root Mean Square Error (RMSE)."""
    obs = obs.flatten()
    sim = sim.flatten()

    mse = np.mean((sim - obs) ** 2)
    rmse_val = np.sqrt(mse)

    return rmse_val


def calc_mae(obs: np.ndarray, sim: np.ndarray):
    """Calculate Mean Absolute Error (MAE)."""
    obs = obs.flatten()
    sim = sim.flatten()

    mae_val = np.mean(abs(sim - obs))

    return mae_val

