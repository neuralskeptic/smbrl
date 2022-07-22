import os

import gpytorch
import torch
from matplotlib import pyplot as plt

from src.utils.seeds import fix_random_seed

seed = 123
fix_random_seed(seed)


def true_f(x):
    # True function is sin(2*pi*x) with Gaussian noise
    return torch.sin(x * (2 * torch.pi)) + torch.randn(x.size()) * torch.sqrt(
        torch.tensor(0.04)
    )


# Training data is 100 points in [0,1] inclusive regularly spaced
train_x = torch.cat(
    [
        torch.distributions.Uniform(0, 0.2).sample((20,)),
        torch.distributions.Uniform(0.6, 0.8).sample((20,)),
    ]
)
test_x = torch.linspace(0, 1, 1000)
train_y = true_f(train_x)
test_y = true_f(test_x)

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.1
)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

n_iter = 50
for i in range(n_iter):
    model.train()
    likelihood.train()
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()
    if i % 2 == 0:
        with torch.no_grad():
            model.eval()
            likelihood.eval()
            print(
                f"Iter %d/%d\t Loss: %.3f\t lengthscale: %.3f\t noise: %.3f"
                "\t train logprob: %.2f\t test logprob: %.2f"
                % (
                    i + 1,
                    n_iter,
                    loss.item(),
                    model.covar_module.base_kernel.lengthscale.item(),
                    model.likelihood.noise.item(),
                    likelihood(model(train_x)).log_prob(train_y),
                    likelihood(model(test_x)).log_prob(test_y),
                )
            )


# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# # make predictions
# f_preds = model(test_x)
# y_preds = likelihood(model(test_x))

# f_mean = f_preds.mean
# f_var = f_preds.variance
# f_covar = f_preds.covariance_matrix
# f_samples = f_preds.sample(sample_shape=torch.Size(1000,))

# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(test_x))

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(test_x, test_y, ".", color="grey")
    ax.plot(train_x, train_y, "k*")
    # Plot predictive means as blue line
    ax.plot(test_x, observed_pred.mean, "b")
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x, lower, upper, alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(["Observed Data", "Mean", "Confidence"])
