import torch
import torch.utils.data as data

def fractions_to_lengths(dataset, fractions):
    lengths = []
    for fraction in fractions:
        lengths.append(int(round(len(dataset) * fraction)))
    
    num_rest = len(dataset) - sum(lengths)
    lengths.append(num_rest)
    return lengths

def classwise_split(dataset, shuffle=True):
    classes = set()
    classwise_indices = dict()
    for index in range(len(dataset)):
        _, label = dataset.__getitem__(index)

        # Should we encounter a new label we add a new bucket
        if label not in classes:
            classes.add(label)
            classwise_indices[label] = []

        # We add the current sample to the bucket of its class
        classwise_indices[label].append(index)

    if shuffle:
        # Torch randperm based shuffle of all buckets
        for key, value in classwise_indices.items():
            classwise_indices[key] = [value[index] for index in iter(torch.randperm(len(value)))]

    return [data.Subset(dataset, classwise_indices[key]) for key in classwise_indices.keys()]

def stratified_split(dataset, lengths, min_num_minority=1):
    total_length = sum(lengths)
    if total_length != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset.")

    if any([length <= 0 for length in lengths]):
        raise ValueError("Any dataset needs to have a length greater zero.")

    classwise_datasets = classwise_split(dataset, shuffle=True)
    num_classes = len(classwise_datasets)
    num_samples_minority = min([len(classwise_dataset) for classwise_dataset in classwise_datasets])

    num_splits = len(lengths)
    if num_samples_minority < num_splits * min_num_minority:
        raise ValueError(
            'The dataset can not be split in {} datasets because the minority class only has {} samples.'.format(
                num_splits, num_samples_minority))

    if any([length < min_num_minority * num_classes for length in lengths]):
        raise ValueError(
            'A minimum number of {} samples for each of the {} classes cannot be guaranteed because the minimum length is {}'.format(
                min_num_minority, num_classes, min(lengths)))

    lengths = [length - num_classes * min_num_minority for length in lengths]
    classwise_lengths = [len(classwise_dataset) - num_splits * min_num_minority for classwise_dataset in
                         classwise_datasets]
    total_samples = sum(classwise_lengths)
    if total_samples == 0:
        fractions = [item for item in [0] for _ in range(num_classes)]
    else:
        fractions = [classwise_length / total_samples for classwise_length in classwise_lengths]

    class_specific_split_datasets = []
    running_lengths = [item for item in [0] for _ in range(num_splits)]
    idx = torch.randperm(num_splits).tolist()
    counter = 0
    for classwise_dataset, fraction in zip(classwise_datasets, fractions):
        list_min_minority = [item for item in [min_num_minority] for _ in range(num_splits)]
        first_split = data.random_split(classwise_dataset,
                                        [len(classwise_dataset) - num_splits * min_num_minority] + list_min_minority)
        classwise_dataset = first_split[0]
        class_specific_single_elements = first_split[1:]

        class_specific_lengths = []
        for length, running_length in zip(lengths, running_lengths):
            class_specific_lengths.append(min(int(length * fraction), length - running_length))
        samples_left = len(classwise_dataset) - sum(class_specific_lengths)

        for i in range(num_splits):
            running_lengths[i] += class_specific_lengths[i]

        while samples_left > 0:
            choice = counter % num_splits
            if running_lengths[idx[choice]] < lengths[idx[choice]]:
                i = idx[choice]
                samples_left -= 1
            else:
                counter += 1
                continue
            class_specific_lengths[i] += 1
            running_lengths[i] += 1
            counter += 1

        second_split = data.random_split(classwise_dataset, class_specific_lengths)
        rejoined_datasets = [data.ConcatDataset([first, second]) for first, second in
                             zip(class_specific_single_elements, second_split)]
        class_specific_split_datasets.append(rejoined_datasets)

    datasets = []
    for i in range(num_splits):
        datasets.append(
            data.ConcatDataset([class_specific_dataset[i] for class_specific_dataset in class_specific_split_datasets]))
    return datasets