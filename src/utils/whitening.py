from functools import wraps

import torch


def whitening_metrics(X, Y):
    # ZCA for X
    x_mean = X.mean(axis=0)  # along N
    x_centered = X - x_mean
    sigma = x_centered.t() @ x_centered / X.shape[0]
    U, S, _ = torch.svd(sigma)
    w_ZCA = U @ torch.diag(torch.sqrt(S + 1e-6) ** -1) @ U.t()
    # z-score for Y
    y_mean = Y.mean(axis=0)  # along N
    y_std = Y.std(axis=0)  # along N
    return x_mean, w_ZCA, y_mean, y_std


def data_whitening(wrapped):
    """
    class decorator to transparently whiten & dewhiten the data fed through
        forward, elbo, ellh
    methods and store the parameters in model (when saved).
    """

    @wraps(wrapped, updated=())
    class _WhiteningWrapper(wrapped):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.register_buffer("do_whitening_x", torch.tensor(True))
            self.register_buffer("do_whitening_y", torch.tensor(True))
            self.register_buffer("x_mean", torch.zeros(self.dim_x))
            self.register_buffer("w_ZCA", torch.eye(self.dim_x, self.dim_x))
            self.register_buffer("y_mean", torch.zeros(self.dim_y))
            self.register_buffer("y_std", torch.ones(self.dim_y))

        def init_whitening(self, X, Y, disable_x=False, disable_y=False):
            self.do_whitening_x = torch.tensor(not disable_x)
            self.do_whitening_y = torch.tensor(not disable_y)
            self.x_mean, self.w_ZCA, self.y_mean, self.y_std = whitening_metrics(X, Y)

        def forward(self, x, *args, **kwargs):
            if self.do_whitening_x:
                x = self.whitenX(x)
            out = super().forward(x, *args, **kwargs)
            if self.do_whitening_y:
                return self.dewhitenY(out)
            return out

        def elbo(self, x, y):
            if self.do_whitening_x:
                x = self.whitenX(x)
            if self.do_whitening_y:
                y = self.whitenY(y)
            return super().elbo(x, y)

        def ellh(self, x, y):
            if self.do_whitening_x:
                x = self.whitenX(x)
            if self.do_whitening_y:
                y = self.whitenY(y)
            return super().ellh(x, y)

        def whitenX(self, x):
            return (x - self.x_mean) @ self.w_ZCA

        def whitenY(self, y):
            return (y - self.y_mean) / self.y_std

        def dewhitenY(self, y):
            return y * self.y_std + self.y_mean

    return _WhiteningWrapper
