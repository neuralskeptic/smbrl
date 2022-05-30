
class SpectralNormalizedNeuralGaussianProcess(LinearBayesianModel):

    d_approx = 1024  # RFFs require ~512-1024 for accuracy

    def __init__(self,  dim_x, dim_y, dim_features):
        self.features = TwoLayerNormalizedResidualNetwork(dim_x, self.d_approx, dim_features)
        super().__init__(dim_x, dim_y, self.d_approx)
