import torch
import torch.nn.functional as F

__all__ = ['matching_loss']

def matching_loss(support_set, query_set, support_targets, query_targets=None, **kwargs):
    """Compute the loss (i.e. negative log-likelihood) for the matching
    network on the test/query points, or logits for query points
    when query targets are not provided.

    Parameters
    ----------
    support_set : `torch.FloatTensor` instance
        A tensor containing the embeddings of the support points. This tensor has
        shape `(batch_size, num_support_examples, embedding_size)`.

    query_set : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has
        shape `(batch_size, num_query_examples, embedding_size)`.

    support_targets : `torch.LongTensor` instance
        A tensor containing the targets of the support points. This tensor has
        shape `(batch_size, num_support_examples)`.

    query_targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has
        shape `(batch_size, num_query_examples)`.

    Returns
    -------
    loss : `torch.FloatTensor` instance
        The negative log-likelihood on the query points when query
        targets are provided.
    logits : `torch.FloatTensor` instance
        A tensor containing logits for the query points over the support
        points when query targets are not provided. This tensor has
        shape `(batch_size, num_query_examples, num_support_examples)`.
    """
    # Similarities Calculation
    # https://github.com/BoyuanJiang/matching-networks-pytorch/blob/a5ea931d67dbf643df8551919cab3af2b5b3e288/matching_networks.py#L80
    eps = 1e-10
    sum_support = torch.sum(torch.pow(support_set, 2), 2)
    support_magnitude = sum_support.clamp(eps, float("inf")).rsqrt()
    dot_product = query_set.bmm(torch.transpose(support_set, 1, 2))
    similarities = dot_product * support_magnitude.unsqueeze(1).repeat(1, dot_product.shape[1], 1)

    # Attention Classification
    # https://github.com/BoyuanJiang/matching-networks-pytorch/blob/a5ea931d67dbf643df8551919cab3af2b5b3e288/matching_networks.py#L63
    softmax_similarities = F.softmax(similarities, dim=2)
    logits = softmax_similarities.bmm(F.one_hot(support_targets).float())

    if query_targets is not None:
        return F.cross_entropy(logits.view(-1, logits.shape[-1]), query_targets.view(-1))
    else:
        return logits
