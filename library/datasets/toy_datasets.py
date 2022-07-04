import torch
from torch.utils.data import Dataset


class Sine1dDataset(Dataset):
    """
    Dataset for creating one-dimensional sinisoidal data with gaussian noise.
    The shape of the data is specified as a list of tuples describing an interval
    and the number of datapoints uniformly sampled from that interval, e.g.:
        [(-1, 0, 100), (0, 1, 50)]
        => 100 data points sampled in the interval [-1, 0] and 50 points in the
           interval (0, 1)
    The x data points are sorted, so the dataset can be easily plotted, so make
    sure to shuffle it before training.
    params:
        data_spec: list of tuples, [(x1a, x1b, N1), (x2a, x2b, N2), ...] (default=[(-1, 1, 100)])
        x_scale: scaling applied to x values when evaluating sine, the x values
            in the dataset are left unchanged (default=torch.pi)
        y_scale: scaling applied to output of sine to create y values (default=1)
        noise: scaling to white (standard) normal gaussian noise (default=1e-2)
    """

    def __init__(
        self,
        data_spec=[
            (-1, 1, 100),
        ],
        x_scale=torch.pi,
        y_scale=1,
        noise=1e-2,
    ):
        super(Dataset, self).__init__()
        x_data_parts = []
        for (xa, xb, N) in data_spec:
            x_data_parts += [torch.distributions.Uniform(xa, xb).sample((N, 1))]
        self.x_data = torch.concat(x_data_parts, axis=0)
        # sort x, so it can be used for testing and fill_between
        self.x_data, _ = self.x_data.sort(dim=0)
        self.y_data = y_scale * torch.sin(self.x_data * x_scale) + noise * torch.randn(
            self.x_data.shape
        )

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, idx):
        return self.x_data[idx, :], self.y_data[idx, :]
