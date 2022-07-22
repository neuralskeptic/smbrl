import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.conversion_utils import df2torch

# class DataFrameDataset(Dataset):
#     """
#     Dataset from pandas dataframe: pass dataframe and x and y labels.
#     """

#     def __init__(self, x_df: pd.DataFrame, y_df: pd.DataFrame, device: str = "cpu"):
#         super(Dataset, self).__init__()
#         self.x_data = df2torch(x_df).to(device)
#         self.y_data = df2torch(y_df).to(device)

#     def __len__(self):
#         return self.x_data.shape[0]

#     def __getitem__(self, idx):
#         return (
#             self.x_data[idx, :],
#             self.y_data[idx, :],
#         )
