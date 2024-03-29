from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm


class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """

    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar
        pbar.update(0)

    def _on_step(self):
        # breakpoint()
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        # self._pbar.update(0)
        # self._pbar.update(1)
        self._pbar.refresh()
        # breakpoint()
        # breakpoint()
        # if (self._pbar.n % 10 == 0):
        # print(self._pbar.n)
        # breakpoint()


# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(
        self,
    ):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps, leave=True)
        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()
