import torch
from check_shape import check_shape


class ReplayBuffer(object):
    """
    stores initial data
    X: torch.tensor (N, dim_state)
    Y: torch.tensor (N, dim_action)
    """

    def __init__(
        self,
        dim_state,
        dim_action,
        batchsize=None,
        shuffling=True,
        max_size=int(1e4),  # lowered from 1e6 (GPU mem leaks)
        device="cpu",
    ):
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.max_size = int(max_size)
        self.batchsize = batchsize if batchsize is not None else self.max_size
        self.shuffling = shuffling
        self.device = device

        self.clear()

    def clear(self):
        self._pos = 0
        self.size = 0
        # store in gpu vs move every batch?
        self._state = torch.empty((self.max_size, self.dim_state)).to(self.device)
        self._action = torch.empty((self.max_size, self.dim_action)).to(self.device)
        self._perm = torch.empty((self.max_size), dtype=torch.long).to(self.device)
        # self._state = torch.empty((self.max_size, self.dim_state), device=self.device)
        # self._action = torch.empty((self.max_size, self.dim_action), device=self.device)
        # self._perm = torch.empty((self.max_size), dtype=torch.long, device=self.device)

    def add(self, new_states, new_actions):
        check_shape([new_states], [("N", self.dim_state)])
        check_shape([new_actions], [("N", self.dim_action)])

        new_states = new_states.to(self.device)
        new_actions = new_actions.to(self.device)

        n = new_states.shape[0]
        l = self.max_size - self._pos  # space until end of buffer (no wraparound)
        if n <= l:  # no wraparound
            self._state[self._pos : self._pos + n, :] = new_states
            self._action[self._pos : self._pos + n, :] = new_actions
        else:
            self._state[self._pos : self._pos + l, :] = new_states[0:l]
            self._action[self._pos : self._pos + l, :] = new_actions[0:l]
            self._state[0 : n - l, :] = new_states[l:]
            self._action[0 : n - l, :] = new_actions[l:]

        self._pos = (self._pos + n) % self.max_size
        self.size = min(self.size + n, self.max_size)

    @property
    def states(self):
        return self._state[0 : self.size]

    @property
    def actions(self):
        return self._action[0 : self.size]

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
        # store in gpu vs move every batch?
        if self.itr < self.size:
            indices = self._perm[0 : self.size][self.itr : self.itr + self.batchsize]
            batch_state = self._state[indices]
            batch_action = self._action[indices]
            # batch_state = self._state[indices]
            # batch_action = self._action[indices]
            self.itr += self.batchsize
            return batch_state, batch_action
        else:
            raise StopIteration
