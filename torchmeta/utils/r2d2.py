import torch
import torch.nn.functional as F
import warnings

from collections import namedtuple
from math import sqrt

__all__ = ['ridge_regression']


_RR_weight_bias = namedtuple('_RR_weight_bias', 'weight bias')

def ridge_regression(embeddings,
                     targets,
                     reg_lambda,
                     num_classes=None,
                     use_woodbury=None,
                     scale=True,
                     bias=True):
    r"""Closed-form solution of a linear function of the embeddings, found 
    using ridge regression.

        W^{*} = argmin_{W} ||XW - Y||^{2} + \lambda ||W||^{2}
              = (X^{T}X + \lambda I)^{-1}X^{T}Y

    Parameters
    ----------
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the support points from a single
        task. This tensor has shape `(num_examples, embedding_size)`.

    targets : `torch.LongTensor` or `torch.FloatTensor` instance
        A tensor containing the targets of the support points from a single
        task. If this tensor is a `torch.LongTensor` instance (i.e.
        classification task), then it has shape `(num_examples,)`. If this
        tensor is a `torch.FloatTensor` instance (i.e. regression task), then
        it has shape `(num_examples, output_size)`.

    reg_lambda : float or `torch.FloatTensor` instance
        The regularization constant lambda in the L2-regularized MSE loss. This
        constant can be a tensor.

    num_classes : int (optional)
        Number of classes in the task. If `None`, then `targets` must be a
        `torch.FloatTensor` instance (i.e. regression task).

    use_woodbury : bool (optional)
        If `True`, then the Woodbury formula is used to solve the ridge
        regression. If `None`, then use the Woodbury formula only in cases
        where the number of samples in the support set is smaller than the
        embedding size (standard in few-shot learning).

    scale : bool (default: `True`)
        If `True`, scales the embeddings and targets by the number of
        support points.

    bias : bool (default: `True`)
        If `True`, fit an affine function and return the weights and bias.

    Returns
    -------
    solution : namedtuple instance
        A namedtuple, with fields `weight` and `bias`, containing the weight
        (and bias, if `bias=True`) of the linear function. If `bias=False`,
        then `solution.bias == None`.

    References
    ----------
    .. [1] Bertinetto L., Henriques J. F., Torr P. H.S., Vedaldi A. (2019).
           Meta-learning with differentiable closed-form solvers. In International
           Conference on Learning Representations (https://arxiv.org/abs/1805.08136)
    """
    num_samples, embedding_size = embeddings.size()
    is_regression_task = targets.dtype.is_floating_point
    if num_classes is None:
        if is_regression_task and (targets.ndim != 2):
            raise ValueError('The `targets` tensor has invalid shape `{0}`. '
                             'This tensor must have shape `(num_examples, '
                             'output_size)`.'.format(targets.shape))

        elif not is_regression_task:
            warnings.warn('The number of classes was not given as an input to '
                          '`ridge_regression`. Using the number of classes '
                          'found in `targets`. If you know the number of '
                          'classes (i.e. N in `N-way` classification), then it '
                          'is recommended to provide it to `ridge_regression`.',
                          stacklevel=2)
            num_classes = -1

    if not is_regression_task:
        targets = (F.one_hot(targets, num_classes=num_classes)
                    .to(dtype=embeddings.dtype))

    if use_woodbury is None:
        use_woodbury = (num_samples <= embedding_size + bias)

    if bias:
        ones = embeddings.new_ones((num_samples, 1))
        embeddings = torch.cat([ones, embeddings], dim=1)

    if scale:
        embeddings = embeddings / sqrt(num_samples)
        targets = targets / sqrt(num_samples)

    if use_woodbury:
        eye = torch.eye(num_samples,
                        dtype=embeddings.dtype,
                        device=embeddings.device)
        A = torch.matmul(embeddings, embeddings.t()) + reg_lambda * eye
        solution = torch.linalg.solve(A, targets)
        weight_bias = torch.matmul(embeddings.t(), solution)
    else:
        eye = torch.eye(embedding_size + bias,
                        dtype=embeddings.dtype,
                        device=embeddings.device)
        A = torch.matmul(embeddings.t(), embeddings) + reg_lambda * eye
        b = torch.matmul(embeddings.t(), targets)
        weight_bias = torch.linalg.solve(A, b)

    if bias:
        return _RR_weight_bias(weight=weight_bias[1:].t(), bias=weight_bias[0])
    else:
        return _RR_weight_bias(weight=weight_bias.t(), bias=None)
