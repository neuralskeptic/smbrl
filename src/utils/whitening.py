import torch


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

        self.x_mean = self.x_mean
        self.w_ZCA = self.w_ZCA

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
