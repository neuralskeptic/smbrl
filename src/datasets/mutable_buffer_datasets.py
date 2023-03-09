from typing import List, Sequence

import torch
from check_shape import check_shape


class ReplayBuffer(object):
    def __init__(
        self,
        dim_list: List[int],
        batchsize=None,
        shuffling=True,
        max_size=int(1e4),  # lowered from 1e6 (GPU mem leaks)
        device="cpu",
    ):
        self.dim_list = dim_list
        self._data = [torch.tensor([]) for _ in dim_list]  # one data tensor per dim
        self.max_size = int(max_size)
        self.batchsize = batchsize if batchsize is not None else self.max_size
        self.shuffling = shuffling
        self.device = device

        self.clear()

    def clear(self):
        self._pos = 0
        self.size = 0
        # store in gpu vs move every batch?
        for i, dim in enumerate(self.dim_list):
            if isinstance(dim, Sequence):
                shape = (self.max_size, *dim)
            else:
                shape = (self.max_size, dim)
            self._data[i] = torch.empty(shape).to(self.device)
        self._perm = torch.empty((self.max_size), dtype=torch.long).to(self.device)

    def add(self, new_list: List[torch.tensor]):
        for new, dim in zip(new_list, self.dim_list):
            if isinstance(dim, Sequence):
                check_shape([new], [("N", *dim)])
            else:
                check_shape([new], [("N", dim)])

        n = new_list[0].shape[0]
        l = self.max_size - self._pos  # space until end of buffer (no wraparound)
        for i, new in enumerate(new_list):
            new = new.to(self.device)
            if n <= l:  # no wraparound
                self._data[i][self._pos : self._pos + n, :] = new
            else:
                self._data[i][self._pos : self._pos + l, :] = new[0:l]
                self._data[i][0 : n - l, :] = new[l:]

        self._pos = (self._pos + n) % self.max_size
        self.size = min(self.size + n, self.max_size)

    @property
    def data(self):
        return [data_i[0 : self.size] for data_i in self._data]

    def __len__(self):
        return self.size

    def __iter__(self):
        self.itr = 0
        if self.shuffling:
            self._perm[0 : self.size] = torch.randperm(self.size)
        else:  # sequential data
            self._perm[0 : self.size] = torch.tensor(range(self.size))
        return self

    def __next__(self):
        # return one batch (or less if at end of buffer)
        if self.itr < self.size:
            indices = self._perm[0 : self.size][self.itr : self.itr + self.batchsize]
            batch_list = [data_i[indices] for data_i in self._data]
            self.itr += self.batchsize
            return batch_list
        else:
            raise StopIteration
