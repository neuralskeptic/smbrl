import torch
import torch.nn as nn


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


class Whitening(object):
    """
    performs ZCA whitening on
    """

    def __init__(self, X, Y, disable_x=False, disable_y=False):
        self._X = X
        self._Y = Y
        self.disable_x = disable_x
        self.disable_y = disable_y

        with torch.no_grad():  # just to be safe
            self._initalize_metrics(X, Y)

    def _initalize_metrics(self, X, Y):
        self.x_mean = X.mean(axis=0)  # along N
        x_centered = X - self.x_mean
        sigma = x_centered.t() @ x_centered / X.shape[0]
        U, S, _ = torch.svd(sigma)
        self.w_ZCA = U @ torch.diag(torch.sqrt(S + 1e-6) ** -1) @ U.t()

        # z-score for Y
        self.y_mean = Y.mean(axis=0)  # along N
        self.y_std = Y.std(axis=0)  # along N

    def whitenX(self, x):
        if self.disable_x:
            return x
        else:
            return (x - self.x_mean.to(x.device)) @ self.w_ZCA.to(x.device)

    def whitenY(self, y):
        if self.disable_y:
            return y
        else:
            return (y - self.y_mean.to(y.device)) / self.y_std.to(y.device)

    def dewhitenY(self, y):
        if self.disable_y:
            return y
        else:
            return y * self.y_std.to(y.device) + self.y_mean.to(y.device)


class WhiteningWrapper(nn.Module):
    def __init__(self, nested: nn.Module):
        super().__init__()
        self.nested = nested  # implicitly in self._modules
        kwargs = {"device": self.nested.device}
        dim_x = list(self.nested.children())[0].in_features
        dim_y = list(self.nested.children())[-1].out_features
        self.register_buffer("x_mean", torch.zeros(dim_x, **kwargs))
        self.register_buffer("w_ZCA", torch.eye(dim_x, **kwargs))
        self.register_buffer("y_mean", torch.zeros(dim_y, **kwargs))
        self.register_buffer("y_std", torch.ones(dim_y, **kwargs))

    def __getattr__(self, name):  # called if attr not found here
        if name == "nested":  # because pytorch wraps all self in a dict
            return self._modules["nested"]
        elif name in self._buffers:
            return self._buffers[name]
        else:
            return getattr(self._modules["nested"], name)

    def init_whitening(self, X, Y):
        self.x_mean, self.w_ZCA, self.y_mean, self.y_std = whitening_metrics(X, Y)

    def forward(self, x):
        x_white = self.whitenX(x)
        out_white = self.nested.forward(x_white)
        return self.dewhitenY(out_white)

    def whitenX(self, x):
        return (x - self.x_mean) @ self.w_ZCA

    def dewhitenY(self, y):
        return y * self.y_std + self.y_mean
