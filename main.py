"""
EA-LSTM training & interpretation
---------------------------------------
This script supports two main modes:
• model_train: Train the EA-LSTM model on meteorological and static data, and evaluate each epoch.
• model_interpret_global: Compute global feature attributions with Integrated Gradients.

Expected Output
---------------
• For model_train:
  - Checkpoints of model weights in runs/<dir_name>/model_weights/
  - A cfg.json file in runs/<dir_name>/ containing the run configuration
  - Console logs with per-epoch loss, NSE, and RMSE

• For model_interpret_global:
  - Temporal contributions of meteorological variables
"""


import argparse
import json
import pickle
import captum.attr as attr
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from corecode.dataset import FluxH5, GlobalH5
from corecode.datautils import rescale_features, calc_nse, calc_rmse
from corecode.ealstm import EALSTM

# --------------------------------------------------------------------------------------
# Device selection
# --------------------------------------------------------------------------------------
# Prefer CUDA if available; otherwise fall back to CPU.
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Global settings for training (merged into argparse config)
GLOBAL_SETTINGS = {
    'clip_norm': True,      # enable gradient clipping
    'clip_value': 1,        # clip threshold
    'dropout': 0.5,         # dropout rate
    'initial_forget_gate_bias': 0,
    'seq_length': 260,
}


def get_args() -> Dict:
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    # "mode" selects the entry point (training or interpretation)
    parser.add_argument('mode', choices=["model_train", "model_interpret_global"])
    # Training arguments
    parser.add_argument('--data_root', type=str, help="Root directory of data")
    parser.add_argument('--dir_name', type=str, help="For train mode. name for run directory.")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Number of learning_rate')
    parser.add_argument('--fold', type=int, help='Number of the folds')
    # Interpretation arguments
    parser.add_argument('--weight', type=str, help='weight of the model')
    parser.add_argument('--interp_data', type=str, help='data for interpretation')
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help="Number of epochs")
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help="Number of batch_size")
    parser.add_argument('--hidden_size',
                        type=int,
                        default=128,
                        help="Number of hidden_size")

    cfg = vars(parser.parse_args())

    # combine global settings with user config
    cfg.update(GLOBAL_SETTINGS)
    cfg["data_root"] = Path(cfg["data_root"])

    return cfg


# --------------------------------------------------------------------------------------
# Helpers for training mode
# --------------------------------------------------------------------------------------

def _setup_model_run(cfg: Dict, run_dir: str) -> Dict:
    # Create folder structure for this run
    cfg['run_dir'] = Path("runs") / run_dir
    if not cfg["run_dir"].is_dir():
        cfg['weight_dir'] = cfg["run_dir"] / 'model_weights'
        cfg["weight_dir"].mkdir(parents=True)
    else:
        raise RuntimeError(f"There is already a folder at {cfg['run_dir']}")
    return cfg


def _prepare_model_data(cfg: Dict) -> Dict:
    """Preprocess training data.

    Parameters
    ----------
    cfg : dict
        Dictionary containing the run config

    Returns
    -------
    dict
        Dictionary containing the updated run config
    """
    # csv file containing the static attributes
    cfg["sta_path"] = 'data/static_final.csv'
    cfg["train_file"] = Path(f'data/Train_Weekly_Stratified/{cfg["fold"]}_train.h5')
    cfg["val_file"] = Path(f'data/Val_Weekly_Stratified/{cfg["fold"]}_val.h5')

    # dump a copy of cfg to run directory
    with (cfg["run_dir"] / 'cfg.json').open('w') as fp:
        temp_cfg = {}
        for key, val in cfg.items():
            if isinstance(val, Path):
                temp_cfg[key] = str(val)
            elif isinstance(val, pd.Timestamp):
                temp_cfg[key] = val.strftime(format="%d%m%Y")
            else:
                temp_cfg[key] = val
        json.dump(temp_cfg, fp, sort_keys=True, indent=4)

    return cfg


class Model(nn.Module):
    """Wrapper class that connects EA-LSTM with fully connected layer"""

    def __init__(self,
                 input_size_dyn: int,
                 input_size_stat: int,
                 hidden_size: int,
                 initial_forget_bias: int = 0,
                 dropout: float = 0.0):
        """Initialize model.

        Parameters
        ----------
        input_size_dyn: int
            Number of dynamic input features.
        input_size_stat: int
            Number of static input features.
        hidden_size: int
            Number of LSTM cells/hidden units.
        initial_forget_bias: int
            Value of the initial forget gate bias.
        dropout: float
            Dropout probability in range(0,1).
        """
        super(Model, self).__init__()
        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.initial_forget_bias = initial_forget_bias
        self.dropout_rate = dropout

        self.lstm = EALSTM(input_size_dyn=input_size_dyn,
                           input_size_stat=input_size_stat,
                           hidden_size=hidden_size,
                           initial_forget_bias=initial_forget_bias)

        self.dropout = nn.Dropout(p=dropout)

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(16, 1)

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor = None) \
            -> torch.Tensor:
        """Run forward pass through the model.

        Parameters
        ----------
        x_d : torch.Tensor
            the dynamic input features of shape [batch, seq_length, n_features]
        x_s : torch.Tensor, optional
            Tensor containing the static catchment characteristics, by default None

        Returns
        -------
        out : torch.Tensor
            the network predictions
        h_n : torch.Tensor
            torch.Tensor containing the hidden states of each time step
        c_n : torch.Tensor
            torch.Tensor containing the cell states of each time step

        Args:
            x_a:
        """
        h_n, c_n = self.lstm(x_d, x_s)
        last_h = self.dropout(h_n[:, -1, :])

        y = self.fc1(last_h)
        y = self.fc2(y)
        y = torch.exp(y)

        return y


# --------------------------------------------------------------------------------------
# Training / evaluation loops
# --------------------------------------------------------------------------------------

def model_train(cfg):
    """Train model.

    Parameters
    ----------
    cfg : Dict
        Dictionary containing the run config
    """
    # create folder structure for this run
    cfg = _setup_model_run(cfg, run_dir=cfg['dir_name'])
    # prepare data for training
    cfg = _prepare_model_data(cfg=cfg)

    # prepare PyTorch DataLoader
    train_ds = FluxH5(h5_file=cfg['train_file'],
                      sta_path=cfg["sta_path"])

    val_ds = FluxH5(h5_file=cfg['val_file'],
                    sta_path=cfg["sta_path"])

    train_loader = DataLoader(train_ds,
                              batch_size=cfg["batch_size"],
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)

    val_loader = DataLoader(val_ds,
                            batch_size=2048,
                            shuffle=False,
                            pin_memory=True)

    # create model and optimizer
    input_size_stat = 26
    input_size_dyn = 5

    model = Model(input_size_dyn=input_size_dyn,
                  input_size_stat=input_size_stat,
                  hidden_size=cfg["hidden_size"],
                  initial_forget_bias=cfg["initial_forget_gate_bias"],
                  dropout=cfg["dropout"]).to(DEVICE)

    # define loss function
    loss_func = nn.MSELoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    for epoch in range(1, cfg["epochs"] + 1):

        loss_set = train_epoch(model, optimizer, loss_func, train_loader, cfg)
        loss_mean = np.mean(loss_set)

        model_path = cfg['weight_dir'] / f"model_epoch{epoch}.pt"
        torch.save(model.state_dict(), str(model_path))
        val_gpp_nse, val_gpp_rmse = eval_epoch(model, val_loader)

        print(f'Epoch {epoch}: Loss:{loss_mean:.3f} val NSE:{val_gpp_nse:.3f} val RMSE:{val_gpp_rmse:.3f}')


def train_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, loss_func: nn.Module,
                loader: DataLoader, cfg: Dict):
    # Train model for a single epoch.
    model.train()
    loss_set = []
    # Iterate in batches over training set
    for data in loader:
        # delete old gradients
        optimizer.zero_grad()
        x, y_gpp, x_s = data
        x, y_gpp, x_s = x.to(DEVICE), y_gpp.to(DEVICE), x_s.to(DEVICE)
        gpp_pred = model(x, x_s[:, 0, :])
        # MSELoss
        loss = loss_func(gpp_pred, y_gpp)
        # calculate gradients
        loss.backward()
        if cfg["clip_norm"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_value"])
        # perform parameter update
        optimizer.step()
        loss_set.append(loss.item())

    return loss_set


def eval_epoch(model: nn.Module, loader: DataLoader):
    # evaluation for one epoch
    model.eval()
    gpp_preds, gpp_obs = None, None
    with torch.no_grad():
        for data in loader:
            x, y_gpp, x_s = data
            x, y_gpp, x_s = x.to(DEVICE), y_gpp.to(DEVICE), x_s.to(DEVICE)
            gpp_p = model(x, x_s[:, 0, :])

            if gpp_preds is None:
                gpp_preds = gpp_p.detach().cpu()
                gpp_obs = y_gpp.detach().cpu()
            else:
                gpp_preds = torch.cat((gpp_preds, gpp_p.detach().cpu()), 0)
                gpp_obs = torch.cat((gpp_obs, y_gpp.detach().cpu()), 0)

        gpp_preds = rescale_features(gpp_preds.numpy(), variable='gpp')
        gpp_obs = rescale_features(gpp_obs.numpy(), variable='gpp')

        gpp_nse = calc_nse(obs=gpp_obs, sim=gpp_preds)
        gpp_rmse = calc_rmse(obs=gpp_obs, sim=gpp_preds)

    return gpp_nse, gpp_rmse


# --------------------------------------------------------------------------------------
# Global interpretation with Integrated Gradients
# --------------------------------------------------------------------------------------

def model_interpret_global(user_cfg: Dict):
    """Run Integrated Gradients to compute temporal contributions of meteorological variables.
    
    Input
    -----
    user_cfg['weight']: path to the model weight
    user_cfg['interp_data']: path to interpretation data
    
    Output
    ------
    Saves results into 'interp.p' pickle file.
    """
    with open(user_cfg["run_dir"] / 'cfg.json', 'r') as fp:
        run_cfg = json.load(fp)

    # create model
    input_size_stat = 26
    input_size_dyn = 5

    model = Model(input_size_dyn=input_size_dyn,
                  input_size_stat=input_size_stat,
                  hidden_size=run_cfg["hidden_size"],
                  dropout=run_cfg["dropout"]).to(DEVICE)

    # load trained model
    model.load_state_dict(torch.load(user_cfg['weight'], map_location=DEVICE))
    model.eval()
    ig = attr.IntegratedGradients(model)
    ds = GlobalH5(h5_file=user_cfg['interp_data'])
    
    # Large batch if memory allows; adjust for your GPU/CPU RAM
    interp_loader = DataLoader(ds,
                               batch_size=2048,
                               shuffle=False,
                               num_workers=0,
                               pin_memory=True)
    
    # Accumulators for concatenating across batches
    attr_dyn, attr_error = None, None
    attr_row, attr_col, attr_date = None, None, None

    for data in tqdm(interp_loader):
        x, x_base, x_s, x_row, x_col, x_date = data
        x, x_base, x_s = x.to(DEVICE), x_base.to(DEVICE), x_s.to(DEVICE)

        attributions_gpp, approximation_error = ig.attribute(inputs=(x, x_s),
                                                             n_steps=3000,
                                                             baselines=(x_base, x_s),
                                                             return_convergence_delta=True)

        if attr_dyn is None:
            attr_dyn = attributions_gpp[0].detach().cpu()
            attr_error = approximation_error.detach().cpu()
            attr_row = x_row
            attr_col = x_col
            attr_date = x_date
        else:
            attr_dyn = torch.cat((attr_dyn, attributions_gpp[0].detach().cpu()), 0)
            attr_error = torch.cat((attr_error, approximation_error.detach().cpu()), 0)
            attr_row = torch.cat((attr_row, x_row), 0)
            attr_col = torch.cat((attr_col, x_col), 0)
            attr_date = torch.cat((attr_date, x_date), 0)

    attr_dyn = attr_dyn.numpy()
    attr_error = attr_error.numpy()
    attr_row = attr_row.numpy()
    attr_col = attr_col.numpy()
    attr_date = attr_date.numpy()

    interp_res = {'dynamic': attr_dyn,
                  'error': attr_error,
                  'row': attr_row,
                  'col': attr_col,
                  'date': attr_date}

    with open('interp.p', "wb") as fp:
        pickle.dump(interp_res, fp)


if __name__ == "__main__":
    config = get_args()
    globals()[config["mode"]](config)



