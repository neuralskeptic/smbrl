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
x0 = np.array([[-4.5, -4.5, np.pi / 2]]).T

dimx = 3
dimu = 2

Q = np.zeros((3, 3))
# Qf = np.eye(dimx)
Qf = np.diag([1, 1, 0])
R = np.diag([0.1, 1])

N = 20


def f(x, u):
    xdot = np.zeros(x.shape)
    xdot[0] = u[0] * np.cos(x[2])
    xdot[1] = u[0] * np.sin(x[2])
    xdot[2] = u[0] * np.tan(u[1])
    return xdot


def f_x(x, u):
    # x = x.reshape(dimx)
    # u = x.reshape(dimu)
    return np.array(
        [[0, 0, -u[0] * np.sin(x[2])], [0, 0, u[0] * np.cos(x[2])], [0, 0, 0]]
    )


def f_u(x, u):
    # x = x.reshape(dimx)
    # u = x.reshape(dimu)
    return np.array(
        [
            [
                np.cos(x[2]),
                0,
            ],
            [
                np.sin(x[2]),
                0,
            ],
            [np.tan(u[1]), u[0] / (np.cos(u[1]) ** 2)],
        ]
    )


def J(u):
    u = u.reshape((dimu, 1, N))

    acc_cost = 0
    x = np.empty([dimx, 1, N + 1])
    x[:, :, 0] = x0
    for t in range(N):  # t = 0,...,N-1
        acc_cost += (x[:, :, t]).T @ Q @ (x[:, :, t])
        acc_cost += u[:, :, t].T @ R @ u[:, :, t]
        x[:, :, t + 1] = f(x[:, :, t], u[:, :, t])
    acc_cost += x[:, :, N].T @ Qf @ x[:, :, N]
    return acc_cost.item(), x


def rollout(u):
    x = np.empty((dimx, 1, N + 1))
    x[:, :, 0] = x0
    dx = np.empty((dimx, 1, N))
    for t in range(N):
        dx[:, :, t] = f(x[:, :, t], u[:, :, t])
        x[:, :, t + 1] = x[:, :, t] + dx[:, :, t]
    return x, dx


### DP solver
def gauss_newton_lqr(u_init):
    u = u_init
    us = []
    for i in range(5):
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

            # P[:, :, t] = P[:, :, t+1]  # init P_t
            # P_ = P[:, :, t+1]  # init P_t
            # # P_last_t = np.zeros(P[:, :, t].shape)
            # i = 0
            # # while not np.allclose(P_last_t, P[:, :, t]):
            # print(np.round(P_, 5))
            # for i in range(10):
            #     i += 1
            #     dP_t = -(Q + At.T @ P_ + P_ @ At.T
            #                 - P_ @ Bt @ np.linalg.inv(R) @ Bt.T @ P_)
            #     # dP_t = -(Q + At.T @ P[:, :, t] + At.T @ P[:, :, t]
            #     #             - P[:, :, t] @ Bt @ np.linalg.inv(R) @ Bt.T @ P[:, :, t])
            #     # P_last_t = P[:, :, t].copy()
            #     P_ += dP_t * 0.1
            #     print(np.round(dP_t, 5))
            #     # print(np.round(At.T @ P_ + P_ @ At, 5))
            #     # print(np.round(- P_ @ Bt @ np.linalg.inv(R) @ Bt.T @ P_, 5))
            #     # print(np.round(P_, 5))
            #     # print(P[:, :, t])
            #     # print(np.allclose(P[:, :, t], P_last_t))
            # P[:, :, t] = P_
            # # breakpoint()
            # # print(i)
            # print(P[:, :, t])
            # print(At.T @ P[:, :, t+1] @ At)
            print(At.T @ P[:, :, t + 1])
            # print(At)
            # print(np.linalg.inv(R + Bt.T @ P[:, :, t+1] @ Bt))
            # print(Bt.T @ P[:, :, t+1] @ At)
            # print(Bt.T @ P[:, :, t+1])
            # print(Bt.T)
            # print(P[:, :, t+1])
        K = np.zeros((dimu, dimx, N))
        x[:, :, 0] = x0
        for t in range(N):
            # K[:, :, t] = - np.linalg.inv(R) @ Bt.T @ P[:, :, t]
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
# u_init = np.random.randn(dimu*N).reshape(dimu, 1, N)
u_init = np.vstack([0.5 * np.ones(N), -0.2 * np.ones(N)]).reshape((dimu, 1, N))
P, K, x, u, us = gauss_newton_lqr(u_init)
x_ = x.reshape((dimx, N + 1))
u_ = u.reshape((dimu, N))

# # steady state of value matrix and gain
# P_ss, K_ss = ss_solve()

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
# for i in range(len(us)):
#     u_ = us[i].reshape(N)
#     axs[0, 0].plot(u_)
axs[0, 0].plot(u_[0, :], c="k", label="u*")
axs[0, 0].legend()
axs[0, 0].set_ylabel("u_s")
axs[1, 0].plot(u_[1, :])
axs[1, 0].set_ylabel("u_phi")
axs[1, 0].set_xlabel("t")
axs[0, 1].plot(x_[0, :], x_[1, :])
axs[0, 1].set_ylabel("y")
axs[0, 1].set_xlabel("x")
axs[1, 1].plot(x_[2, :])
axs[1, 1].set_ylabel("phi")
axs[1, 1].set_xlabel("t")
for ax in axs.flatten():
    ax.grid(True)
fig.suptitle(f"optimal controller (cost={234})")

# plt.figure()
# plt.plot(K[0, 0, :])
# plt.plot(K[0, 1, :])
# plt.grid(True)
