import torch
import numpy as np


class NumpyToTorch:
    """Convert a numpy.ndarray to a pytorch.tensor."""

    def __call__(self, numpy_array: np.ndarray) -> torch.tensor:
        """
        Parameters
        ----------
        numpy_array : np.ndarray
            the numpy array

        Returns
        -------
        torch.tensor
            converted torch array with the same values as the numpy array
        """
        return torch.from_numpy(numpy_array).contiguous()

    def __repr__(self):
        return self.__class__.__name__ + '()'
