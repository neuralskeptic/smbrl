import torch
from check_shape import check_shape


class ReplayBuffer(object):
    def __init__(
        self,
        dim_x,
        dim_y,
        batchsize=None,
        shuffling=True,
        max_size=int(1e4),  # lowered from 1e6 (GPU mem leaks)
        device="cpu",
    ):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.max_size = int(max_size)
        self.batchsize = batchsize if batchsize is not None else self.max_size
        self.shuffling = shuffling
        self.device = device

        self.clear()

    def clear(self):
        self._pos = 0
        self.size = 0
        # store in gpu vs move every batch?
        self._x = torch.empty((self.max_size, self.dim_x)).to(self.device)
        self._y = torch.empty((self.max_size, self.dim_y)).to(self.device)
        self._perm = torch.empty((self.max_size), dtype=torch.long).to(self.device)
        # self._x = torch.empty((self.max_size, self.dim_x), device=self.device)
        # self._y = torch.empty((self.max_size, self.dim_y), device=self.device)
        # self._perm = torch.empty((self.max_size), dtype=torch.long, device=self.device)

    def add(self, new_xs, new_ys):
        check_shape([new_xs], [("N", self.dim_x)])
        check_shape([new_ys], [("N", self.dim_y)])

        new_xs = new_xs.to(self.device)
        new_ys = new_ys.to(self.device)

        n = new_xs.shape[0]
        l = self.max_size - self._pos  # space until end of buffer (no wraparound)
        if n <= l:  # no wraparound
            self._x[self._pos : self._pos + n, :] = new_xs
            self._y[self._pos : self._pos + n, :] = new_ys
        else:
            self._x[self._pos : self._pos + l, :] = new_xs[0:l]
            self._y[self._pos : self._pos + l, :] = new_ys[0:l]
            self._x[0 : n - l, :] = new_xs[l:]
            self._y[0 : n - l, :] = new_ys[l:]

        self._pos = (self._pos + n) % self.max_size
        self.size = min(self.size + n, self.max_size)

    @property
    def xs(self):
        return self._x[0 : self.size]

    @property
    def ys(self):
        return self._y[0 : self.size]

    def __len__(self):
        return self.size

    def __iter__(self):
        self.itr = 0
        if self.shuffling:
            self._perm[0 : self.size] = torch.randperm(self.size)  # shuffle data
        else:
            self._perm[0 : self.size] = torch.tensor(
                range(self.size)
            )  # sequential data
        return self

    def __next__(self):
        # return one batch (or less if at end of buffer)
        if self.itr < self.size:
            indices = self._perm[0 : self.size][self.itr : self.itr + self.batchsize]
            batch_x = self._x[indices]
            batch_y = self._y[indices]
            # batch_x = self._x[indices]
            # batch_y = self._y[indices]
            self.itr += self.batchsize
            return batch_x, batch_y
        else:
            raise StopIteration
