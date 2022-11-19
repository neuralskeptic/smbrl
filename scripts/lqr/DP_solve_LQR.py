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
x0 = np.array([[1, 0]]).T
A = np.array([[1, 1], [0, 1]])
B = np.array([[0, 1]]).T

C = np.array([[1, 0]])
rho = 0.3  # TUNE ME
# rho = 10  # TUNE ME

Q = C.T @ C
# Qf = Q
Qf = np.zeros(Q.shape)
# Qf = 1e3 * np.eye(2)
R = rho * np.eye(B.shape[1])

N = 20


def f(x, u):
    xdot = np.einsum("...x,x...n->...n", A, x)
    xdot += np.einsum("xu,u...n->x...n", B, u)
    return xdot


def g(x):
    return np.einsum("yx,x...n->y...n", C, x)  # y


def J(u):
    u = u.reshape((B.shape[1], x0.shape[1], N))

    acc_cost = 0
    x = np.empty([x0.shape[0], x0.shape[1], N + 1])
    x[:, :, 0] = x0
    for t in range(N):  # t = 0,...,N-1
        acc_cost += x[:, :, t].T @ Q @ x[:, :, t]
        acc_cost += u[:, :, t].T @ R @ u[:, :, t]
        x[:, :, t + 1] = f(x[:, :, t], u[:, :, t])
        # breakpoint()
    acc_cost += x[:, :, N].T @ Qf @ x[:, :, N]
    return acc_cost.item(), x


### DP solver
def dp_solve():
    P = np.empty((Q.shape[0], Q.shape[1], N + 1))
    P[:, :, N] = Qf
    for t in np.arange(N)[::-1] + 1:  # t = N,...,1
        P[:, :, t - 1] = (
            Q
            + A.T @ P[:, :, t] @ A
            - A.T
            @ P[:, :, t]
            @ B
            @ (R + B.T @ P[:, :, t] @ B) ** (-1)
            @ B.T
            @ P[:, :, t]
            @ A
        )
    K = np.empty((B.shape[1], x0.shape[0], N))
    u = np.empty((B.shape[1], x0.shape[1], N))
    x = np.empty((x0.shape[0], x0.shape[1], N + 1))
    x[:, :, 0] = x0
    for t in range(N):
        K[:, :, t] = -((R + B.T @ P[:, :, t] @ B) ** (-1)) @ B.T @ P[:, :, t] @ A
        u[:, :, t] = K[:, :, t] @ x[:, :, t]
        x[:, :, t + 1] = f(x[:, :, t], u[:, :, t])
    return P, K, x, u


def ss_solve():
    m = 1e3
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


### dummy controller
dummy_u = -0.01 * np.ones(N)
cost, x = J(dummy_u)
y = g(x).reshape(N + 1)
x_ = x.reshape(-1, N + 1)
u_ = dummy_u.reshape(N)

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0, 0].plot(u_)
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
fig.suptitle(f"dummy controller (cost={cost})")


### optimal controller
P, K, x, u = dp_solve()
x_ = x.reshape(-1, N + 1)
u_ = u.reshape(N)
y = g(x_).reshape(N + 1)

# steady state of value matrix and gain
P_ss, K_ss = ss_solve()

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0, 0].plot(u_)
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
fig.suptitle(f"optimal controller (cost={234})")

plt.figure()
plt.plot(K[0, 0, :])
plt.plot(K[0, 1, :])
plt.grid(True)
