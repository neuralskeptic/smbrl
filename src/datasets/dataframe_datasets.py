import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.conversion_utils import df2torch


class DataFrameDataset(Dataset):
    """
    Dataset from pandas dataframe: pass dataframe and x and y labels.
    """

    def __init__(self, x_df: pd.DataFrame, y_df: pd.DataFrame):
        super(Dataset, self).__init__()
        self.x_data = x_df
        self.y_data = y_df

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, idx):
        return (
            df2torch(self.x_data.iloc[idx, :]),
            df2torch(self.y_data.iloc[idx, :]),
        )
