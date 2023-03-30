import torch

from src.utils.torch_tools import NoTraining, Stateless


class Pendulum(Stateless, NoTraining):

    dim_x = 2
    dim_u = 1
    dim_xu = 3
    u_mx = 2.0

    def __call__(self, xu: torch.Tensor) -> torch.Tensor:
        dt = 0.05
        m = 1.0
        l = 1.0
        d = 1e-2  # damping
        g = 9.80665
        x, u = xu[..., :2], xu[..., 2]
        u = torch.clip(u, -self.u_mx, self.u_mx)
        th_dot_dot = (
            -3.0 * g / (2 * l) * torch.sin(x[..., 0] + torch.pi) - d * x[..., 1]
        )
        th_dot_dot += 3.0 / (m * l**2) * u
        # variant from http://underactuated.mit.edu/pend.html
        # th_dot_dot = -g / l * torch.sin(x[:, 0] + torch.pi) - d / (m*l**2) * x[:, 1]
        # th_dot_dot += 1.0 / (m * l**2) * u
        x_dot = x[..., 1] + th_dot_dot * dt  # theta_dot
        x_pos = x[..., 0] + x_dot * dt  # theta

        x2 = torch.stack((x_pos, x_dot), dim=-1)
        xu[..., 2] = u
        return x2, xu

    def run(self, initial_state, policy, horizon):
        batch_shape = [horizon] + list(initial_state.shape[:-1])
        xs = torch.zeros(batch_shape + [self.dim_x])
        us = torch.zeros(batch_shape + [self.dim_u])
        xxs = torch.zeros(batch_shape + [self.dim_x])
        state = initial_state
        for t in range(horizon):
            action = policy.predict(state, t=t)
            xu = torch.cat((state, action), dim=-1)
            xxs[t, ...], _ = self.__call__(xu)
            xs[t, ...] = state
            us[t, ...] = action
            state = xxs[t, ...]
        return xs, us, xxs

    def cost(self, x, **kw):
        """batched version (batch dims first)"""
        # swing-up: \theta = \pi -> 0
        theta, theta_dot, u = x[..., 0], x[..., 1], x[..., 2]
        theta_cost = (torch.cos(theta) - 1.0) ** 2
        theta_dot_cost = 1e-2 * theta_dot**2
        u_cost = 1e-2 * u**2
        total_cost = theta_cost + theta_dot_cost + u_cost
        return total_cost
