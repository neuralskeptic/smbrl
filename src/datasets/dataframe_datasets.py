import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class DataFrameDataset(Dataset):
    """
    Dataset from pandas dataframe: pass dataframe and x and y labels.
    If no seed is passed, seed=0 is used, to keep the train-test split disjoint
    between the train and the test dataset creation.
    """

    def __init__(self, x_df: pd.DataFrame, y_df: pd.DataFrame, train=True, seed=0):
        super(Dataset, self).__init__()
        x_train, x_test, y_train, y_test = train_test_split(
            x_df, y_df, test_size=0.2, random_state=seed
        )

        if train:
            self.x_data, self.y_data = x_train, y_train
        else:
            self.x_data, self.y_data = x_test, y_test

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x_data.iloc[idx, :].values),
            torch.tensor(self.y_data.iloc[idx, :].values),
        )
