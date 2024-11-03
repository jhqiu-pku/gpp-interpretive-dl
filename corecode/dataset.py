import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from corecode.datautils import load_attributes


class FluxH5(Dataset):
    def __init__(self,
                 h5_file,
                 sta_path: str,
                 cache: bool = False):

        self.h5_file = h5_file
        self.sta_path = sta_path
        self.cache = cache
        self.df = None
        self.attribute_means = None
        self.attribute_stds = None

        # preload data if cached is true
        if self.cache:
            (self.x, self.y_gpp, self.site_id) = self._preload_data()

        # load attributes into data frame
        self._load_attributes()

        # determine number of samples once
        if self.cache:
            self.num_samples = self.y_gpp.shape[0]
        else:
            with h5py.File(h5_file, 'r') as f:
                self.num_samples = f["output_gpp"].shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        if self.cache:
            x = self.x[idx]
            y_gpp = self.y_gpp[idx]
            site = self.site_id[idx]
        else:
            with h5py.File(self.h5_file, 'r') as f:
                x = f["weekly_data"][idx]
                y_gpp = f["output_gpp"][idx]
                site = f["site_id"][idx]
                site = site.decode("ascii")

        attributes = self.df.loc[self.df.index == site].values
        # convert to torch tensors
        attributes = torch.from_numpy(attributes.astype(np.float32))
        x = torch.from_numpy(x.astype(np.float32))
        y_gpp = torch.from_numpy(y_gpp.astype(np.float32))

        return x, y_gpp, attributes

    def _preload_data(self):
        with h5py.File(self.h5_file, 'r') as f:
            x = f["weekly_data"][:]
            y_gpp = f["output_gpp"][:]
            str_arr = f["site_id"][:]
            str_arr = [x.decode("ascii") for x in str_arr]

        return x, y_gpp, str_arr

    def _load_attributes(self):
        df = load_attributes(self.sta_path)
        df_igbp = df['IGBP']
        df_igbp = pd.get_dummies(df_igbp)
        df_num = df.drop('IGBP', axis=1)
        # store means and stds
        self.attribute_means = df_num.mean()
        self.attribute_stds = df_num.std()
        # normalize data
        df_num = (df_num - self.attribute_means) / self.attribute_stds
        df = pd.concat((df_num, df_igbp), axis=1)
        self.df = df


class GlobalH5(Dataset):
    def __init__(self,
                 h5_file):

        self.h5_file = h5_file

        # preload data if cached is true
        (self.x, self.x_base, self.x_static, self.x_row, self.x_col, self.x_date) = self._preload_data()
        # determine number of samples once
        self.num_samples = self.x.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):

        x = self.x[idx]
        x_base = self.x_base[idx]
        x_static = self.x_static[idx]
        x_row = self.x_row[idx]
        x_col = self.x_col[idx]
        x_date = self.x_date[idx]

        # convert to torch tensors
        x = torch.from_numpy(x.astype(np.float32))
        x_base = torch.from_numpy(x_base.astype(np.float32))
        x_static = torch.from_numpy(x_static.astype(np.float32))

        return x, x_base, x_static, x_row, x_col, x_date

    def _preload_data(self):
        with h5py.File(self.h5_file, 'r') as f:
            x = f["weekly_data"][:]
            x_base = f["base_data"][:]
            x_static = f["static_data"][:]
            x_row = f["pixel_row"][:]
            x_col = f["pixel_col"][:]
            x_date = f["pixel_date"][:]

        return x, x_base, x_static, x_row, x_col, x_date
