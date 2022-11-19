# Problem:
# min_U J(U) = x_N'Q_fx_N + \Sigma^{N-1}_{t=0} x_t'Qx_t + u_t'Ru_t
# s.t. x_t+1 = Ax_t + Bu_t
# where U := (u_0, ..., u_N-1)
#       N := 20
#       x_0 := [1, 0]
#       A := [[1, 1], [0, 1]]
#       B := [1, 0]'  # typo in problem: first dimension is xdot !
#       Q := Q_f := C'C
#       C := [1, 0]
#       R := rho*I, rho \in [0, \inf)
# also: y_t := Cx_t

import matplotlib.pyplot as plt
import numpy as np

### problem definition
x0 = np.array([[0, 0]]).T
xd = np.array([[np.pi, 0]]).T
m = 5
l = 1
b = 1

dimx = 2
dimu = 1

C = np.array([[1, 0]])
rho = 0.3  # TUNE ME
# rho = 10  # TUNE ME
dt = 0.01

Q = np.zeros((dimx, dimx))
# Qf = np.eye(dimx)
Qf = np.diag([1, 1])
# Qf = np.zeros(Q.shape)
# Qf = 1e3 * np.eye(2)
R = 10 * np.eye(1)

N = 20


def f(x, u):
    xdot = np.zeros(x.shape)
    xdot[0] = x[1]
    xdot[1] = -10 / l * np.sin(x[0]) - b / (m * l**2) * x[1] + u[0] / (m * l**2)
    return xdot


def f_x(x, u):
    return np.array([[1, 0], [-b / m / l, -10 * np.cos(x[0])]])


def f_u(x, u):
    return np.array([[0], [1 / m / l]])


def g(x):
    return np.einsum("yx,x...n->y...n", C, x)  # y


def J(u):
    u = u.reshape((dimu, 1, N))

    acc_cost = 0
    x = np.empty([dimx, 1, N + 1])
    x[:, :, 0] = x0
    for t in range(N):  # t = 0,...,N-1
        acc_cost += (x[:, :, t] - xd).T @ Q @ (x[:, :, t] - xd)
        acc_cost += u[:, :, t].T @ R @ u[:, :, t]
        x[:, :, t + 1] = f(x[:, :, t], u[:, :, t])
        # breakpoint()
    acc_cost += x[:, :, N].T @ Qf @ x[:, :, N]
    return acc_cost.item(), x


def rollout(u):
    x = np.empty((dimx, 1, N + 1))
    x[:, :, 0] = x0
    dx = np.empty((dimx, 1, N))
    for t in range(N):
        dx[:, :, t] = f(x[:, :, t], u[:, :, t]) * 0.001
        x[:, :, t + 1] = x[:, :, t] + dx[:, :, t]
    return x, dx


### DP solver
def gauss_newton_lqr(u_init):
    u = u_init
    us = []
    for i in range(100):
        # roll out u
        x, dx = rollout(u)
        # backprop through time (DARE)
        P = np.zeros((dimx, dimx, N + 1))
        A = np.zeros((dimx, dimx, N))
        B = np.zeros((dimx, dimu, N))
        P[:, :, N] = Qf
        for t in np.arange(N)[::-1]:  # t = N-1,...,0
            A[:, :, t] = f_x(x[:, :, t], u[:, :, t])
            B[:, :, t] = f_u(x[:, :, t], u[:, :, t])
            At = A[:, :, t]
            Bt = B[:, :, t]
            # print(np.linalg.inv(At))
            P[:, :, t] = (
                Q
                + At.T @ P[:, :, t + 1] @ At
                - At.T
                @ P[:, :, t + 1]
                @ Bt
                @ np.linalg.inv(R + Bt.T @ P[:, :, t + 1] @ Bt)
                @ Bt.T
                @ P[:, :, t + 1]
                @ At
            )
            print(P[:, :, t])
            # print(np.linalg.inv(R + Bt.T @ P[:, :, t+1] @ Bt))
            # print(Bt.T @ P[:, :, t+1] @ At)
            # print(Bt.T)
            # print(P[:, :, t+1])
        K = np.zeros((dimu, dimx, N))
        x[:, :, 0] = x0
        for t in range(N):
            K[:, :, t] = (
                -np.linalg.inv(R + Bt.T @ P[:, :, t] @ Bt) @ Bt.T @ P[:, :, t] @ At
            )
            du = K[:, :, t] @ dx[:, :, t]
            # print(du)
            # print(R)
            # print(P[:, :, t])
            u[:, :, t] += du
        us.append(u.copy())
    return P, K, x, u, us


def ss_solve(A, B):
    P = np.eye(2)
    P_last = np.zeros((2, 2))
    i = 0
    while not np.allclose(P, P_last):
        i += 1
        P_last = P
        P = Q + A.T @ P @ A - A.T @ P @ B @ (R + B.T @ P @ B) ** (-1) @ B.T @ P @ A
    K = -((R + B.T @ P @ B) ** (-1)) @ B.T @ P @ A
    # print(i)  # how many iterations to converge
    return P, K


### dimummy controller
# dimummy_u = np.zeros(N)
# dimummy_u[0] = 0.01
# cost, x = J(dimummy_u)
# y = g(x).reshape(N+1)
# x_ = x.reshape(-1, N+1)
# u_ = dimummy_u.reshape(N)

# fig, axs = plt.subplots(2, 2, figsize=(10,6))
# axs[0, 0].plot(u_)
# axs[0, 0].set_ylabel('u')
# axs[1, 0].plot(y[:])
# axs[1, 0].set_ylabel('y')
# axs[1, 0].set_xlabel('t')
# axs[0, 1].plot(x_[0, :])
# axs[0, 1].set_ylabel('x[0]')
# axs[1, 1].plot(x_[1, :])
# axs[1, 1].set_ylabel('x[1]')
# axs[1, 1].set_xlabel('t')
# for ax in axs.flatten():
#     ax.grid(True)
# fig.suptitle(f"dummy controller (cost={cost})")


### optimal controller
u_init = np.random.randn(N).reshape(dimu, 1, N)
P, K, x, u, us = gauss_newton_lqr(u_init)
cost, _ = J(u)
x_ = x.reshape(-1, N + 1)
u_ = u.reshape(N)
y = g(x_).reshape(N + 1)

# # steady state of value matrix and gain
# P_ss, K_ss = ss_solve()

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
for i in range(len(us)):
    u_ = us[i].reshape(N)
    axs[0, 0].plot(u_)
axs[0, 0].plot(u_, c="k", label="u*")
axs[0, 0].legend()
axs[0, 0].set_ylabel("u")
axs[1, 0].plot(y[:])
axs[1, 0].set_ylabel("y")
axs[1, 0].set_xlabel("t")
axs[0, 1].plot(x_[0, :])
axs[0, 1].set_ylabel("x[0]")
axs[1, 1].plot(x_[1, :])
axs[1, 1].set_ylabel("x[1]")
axs[1, 1].set_xlabel("t")
for ax in axs.flatten():
    ax.grid(True)
fig.suptitle(f"optimal controller (cost={cost})")

# plt.figure()
# plt.plot(K[0, 0, :])
# plt.plot(K[0, 1, :])
# plt.grid(True)
