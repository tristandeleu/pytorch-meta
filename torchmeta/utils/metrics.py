import torch
import torch.nn.functional as F

from torchmeta.utils.prototype import get_prototypes

__all__ = ['hardness_metric']


def _pad_images(inputs, size=(224, 224), **kwargs):
    height, width = inputs.shape[-2:]
    pad_height, pad_width = (size[0] - height) // 2, (size[1] - width) // 2
    padding = (pad_width, size[1] - width - pad_width,
               pad_height, size[0] - height - pad_height)
    return F.pad(inputs, padding, **kwargs)


def hardness_metric(batch, num_classes):
    """Hardness metric of an episode, as defined in [1].

    Parameters
    ----------
    batch : dict
        The batch of tasks over which the metric is computed. The batch of tasks
        is a dictionary containing the keys `train` (or `support`) and `test`
        (or `query`). This is typically the output of `BatchMetaDataLoader`.

    num_classes : int
        The number of classes in the classification task. This corresponds to
        the number of ways in an `N`-way classification problem.

    Returns
    -------
    metric : `torch.FloatTensor` instance
        Values of the hardness metric for each task in the batch.

    References
    ----------
    .. [1] Dhillon, G. S., Chaudhari, P., Ravichandran, A. and Soatto S. (2019).
           A Baseline for Few-Shot Image Classification. (https://arxiv.org/abs/1909.02729)
    """
    if ('train' not in batch) and ('support' not in batch):
        raise ValueError('The tasks do not contain any training/support set. '
                         'Make sure the tasks contain either the "train" or the '
                         '"support" key.')
    if ('test' not in batch) and ('query' not in batch):
        raise ValueError('The tasks do not contain any test/query set. Make '
                         'sure the tasks contain either the "test" of the '
                         '"query" key.')

    train = 'train' if ('train' in batch) else 'support'
    test = 'test' if ('test' in batch) else 'query'

    with torch.no_grad():
        # Load a pre-trained backbone Resnet-152 model from PyTorch Hub
        backbone = torch.hub.load('pytorch/vision:v0.5.0',
                                  'resnet152',
                                  pretrained=True,
                                  verbose=False)
        backbone.eval()

        train_inputs, train_targets = batch[train]
        test_inputs, test_targets = batch[test]
        batch_size, num_images, num_channels = train_inputs.shape[:3]
        num_test_images = test_inputs.size(1)

        backbone.to(device=train_inputs.device)

        if num_channels != 3:
            raise ValueError('The images must be RGB images.')

        # Pad the images so that they are compatible with the pre-trained model
        padded_train_inputs = _pad_images(train_inputs,
            size=(224, 224), mode='constant', value=0.)
        padded_test_inputs = _pad_images(test_inputs,
            size=(224, 224), mode='constant', value=0.)

        # Compute the features from the logits returned by the pre-trained
        # model on the train/support examples. These features are z(x, theta)_+,
        # averaged for each class
        train_logits = backbone(padded_train_inputs.view(-1, 3, 224, 224))
        train_logits = F.relu(train_logits.view(batch_size, num_images, -1))
        train_features = get_prototypes(train_logits, train_targets, num_classes)

        # Get the weights by normalizing the features
        weights = F.normalize(train_features, p=2, dim=2)

        # Compute and normalize the logits of the test/query examples
        test_logits = backbone(padded_test_inputs.view(-1, 3, 224, 224))
        test_logits = test_logits.view(batch_size, num_test_images, -1)
        test_logits = F.normalize(test_logits, p=2, dim=2)

        # Compute the log probabilities of the test/query examples
        test_logits = torch.bmm(weights, test_logits.transpose(1, 2))
        test_log_probas = -F.cross_entropy(test_logits, test_targets,
                                           reduction='none')

        # Compute the log-odds ratios for each image of the test/query set
        log_odds_ratios = torch.log1p(-test_log_probas.exp()) - test_log_probas

    return torch.mean(log_odds_ratios, dim=1)
