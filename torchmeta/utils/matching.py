import torch
import torch.nn.functional as F

__all__ = [
    'pairwise_cosine_similarity',
    'matching_log_probas',
    'matching_probas',
    'matching_loss'
]


def pairwise_cosine_similarity(embeddings1, embeddings2, eps=1e-8):
    r"""Computes the pairwise cosine similarity between two tensors of embeddings.

    Parameters
    ----------
    embeddings1 : `torch.Tensor` instance
        A tensor containing embeddings with shape
        `(batch_size, N, embedding_size)`.

    embeddings2 : `torch.Tensor` instance
        A tensor containing embeddings with shape
        `(batch_size, M, embedding_size)`.

    eps: float (default: 1e-8)
        Small value to avoid division by zero.

    Returns
    -------
    similarities : `torch.Tensor` instance
        A tensor containing the pairwise cosine similarities between the vectors
        in `embeddings1` and `embeddings2`. This tensor has shape
        `(batch_size, N, M)`.

    Notes
    -----
    The cosine similarity is computed as

        .. math ::
            \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}
    """
    sq_norm1 = torch.sum(embeddings1 ** 2, dim=2, keepdim=True)
    sq_norm2 = torch.sum(embeddings2 ** 2, dim=2).unsqueeze(1)
    dot_product = torch.bmm(embeddings1, embeddings2.transpose(1, 2))
    inverse_norm = torch.rsqrt(torch.clamp(sq_norm1 * sq_norm2, min=eps ** 2))
    return dot_product * inverse_norm


def matching_log_probas(embeddings, targets, test_embeddings, num_classes, eps=1e-8):
    """Computes the log-probability of test samples given the training dataset
    for the matching network [1].

    Parameters
    ----------
    embeddings : `torch.Tensor` instance
        A tensor containing the embeddings of the train/support inputs. This
        tensor has shape `(batch_size, num_train_samples, embedding_size)`.

    targets : `torch.LongTensor` instance
        A tensor containing the targets of the train/support dataset. This tensor
        has shape `(batch_size, num_train_samples)`.

    test_embeddings : `torch.Tensor` instance
        A tensor containing the embeddings of the test/query inputs. This tensor
        has shape `(batch_size, num_test_samples, embedding_size)`.

    num_classes : int
        Number of classes (i.e. `N` in "N-way classification") in the
        classification task.

    eps : float (default: 1e-8)
        Small value to avoid division by zero.

    Returns
    -------
    log_probas : `torch.Tensor` instance
        A tensor containing the log-probabilities of the test samples given the
        training dataset for the matching network. This tensor has shape
        `(batch_size, num_classes, num_test_samples)`.

    References
    ----------
    .. [1] Vinyals, O., Blundell, C., Lillicrap, T. and Wierstra, D. (2016).
           Matching Networks for One Shot Learning. In Advances in Neural
           Information Processing Systems (pp. 3630-3638) (https://arxiv.org/abs/1606.04080)
    """
    batch_size, num_samples, _ = test_embeddings.shape
    similarities = pairwise_cosine_similarity(embeddings, test_embeddings, eps=eps)
    logsumexp = torch.logsumexp(similarities, dim=1, keepdim=True)

    max_similarities, _ = torch.max(similarities, dim=1, keepdim=True)
    exp_similarities = torch.exp(similarities - max_similarities)

    sum_exp = exp_similarities.new_zeros((batch_size, num_classes, num_samples))
    indices = targets.unsqueeze(-1).expand_as(exp_similarities)
    sum_exp.scatter_add_(1, indices, exp_similarities)

    return torch.log(sum_exp) + max_similarities - logsumexp


def matching_probas(embeddings, targets, test_embeddings, num_classes, eps=1e-8):
    """Computes the probability of test samples given the training dataset for
    the matching network [1].

    Parameters
    ----------
    embeddings : `torch.Tensor` instance
        A tensor containing the embeddings of the train/support inputs. This
        tensor has shape `(batch_size, num_train_samples, embedding_size)`.

    targets : `torch.LongTensor` instance
        A tensor containing the targets of the train/support dataset. This tensor
        has shape `(batch_size, num_train_samples)`.

    test_embeddings : `torch.Tensor` instance
        A tensor containing the embeddings of the test/query inputs. This tensor
        has shape `(batch_size, num_test_samples, embedding_size)`.

    num_classes : int
        Number of classes (i.e. `N` in "N-way classification") in the
        classification task.

    eps : float (default: 1e-8)
        Small value to avoid division by zero.

    Returns
    -------
    probas : `torch.Tensor` instance
        A tensor containing the probabilities of the test samples given the
        training dataset for the matching network. This tensor has shape
        `(batch_size, num_classes, num_test_samples)`.

    References
    ----------
    .. [1] Vinyals, O., Blundell, C., Lillicrap, T. and Wierstra, D. (2016).
           Matching Networks for One Shot Learning. In Advances in Neural
           Information Processing Systems (pp. 3630-3638) (https://arxiv.org/abs/1606.04080)
    """
    log_probas = matching_log_probas(embeddings,
                                     targets,
                                     test_embeddings,
                                     num_classes,
                                     eps=eps)
    return log_probas.exp()


def matching_loss(train_embeddings,
                  train_targets,
                  test_embeddings,
                  test_targets,
                  num_classes,
                  eps=1e-8,
                  **kwargs):
    """Compute the loss (i.e. negative log-likelihood) for the matching network
    on the test/query samples [1].

    Parameters
    ----------
    train_embeddings : `torch.Tensor` instance
        A tensor containing the embeddings of the train/support inputs. This
        tensor has shape `(batch_size, num_train_samples, embedding_size)`.

    train_targets : `torch.LongTensor` instance
        A tensor containing the targets of the train/support dataset. This tensor
        has shape `(batch_size, num_train_samples)`.

    test_embeddings : `torch.Tensor` instance
        A tensor containing the embeddings of the test/query inputs. This tensor
        has shape `(batch_size, num_test_samples, embedding_size)`.

    test_targets : `torch.LongTensor` instance
        A tensor containing the targets of the test/query dataset. This tensor
        has shape `(batch_size, num_test_samples)`.

    num_classes : int
        Number of classes (i.e. `N` in "N-way classification") in the
        classification task.

    eps : float (default: 1e-8)
        Small value to avoid division by zero.

    kwargs :
        Additional keyword arguments to be forwarded to the loss function. See
        `torch.nn.functional.cross_entropy` for details.

    Returns
    -------
    loss : `torch.Tensor` instance
        A tensor containing the loss for the matching network.

    References
    ----------
    .. [1] Vinyals, O., Blundell, C., Lillicrap, T. and Wierstra, D. (2016).
           Matching Networks for One Shot Learning. In Advances in Neural
           Information Processing Systems (pp. 3630-3638) (https://arxiv.org/abs/1606.04080)
    """
    logits = matching_log_probas(train_embeddings,
                                 train_targets,
                                 test_embeddings,
                                 num_classes,
                                 eps=eps)
    return F.nll_loss(logits, test_targets, **kwargs)
