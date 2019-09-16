## Harmonic

Simple regression task, based on the sum of two sine waves, as introduced in [1].

```python
torchmeta.toy.Harmonic(num_samples_per_task, num_tasks=5000, noise_std=None,
    transform=None, target_transform=None, dataset_transform=None)
```

**Parameters**

 - **num_samples_per_task**: *int*
 Number of examples per task.

 - **num_tasks**: *int (default: 5,000)*
 Overall number of tasks to sample.

 - **noise_std**: *float, optional*
 Amount of noise to include in the targets for each task. If `None`, then nos noise is included, and the target is the sum of 2 sine functions of the input.

 - **transform**: *callable, optional*
 A function/transform that takes a numpy array of size (1,) and returns a transformed version of the input.

 - **target_transform**: *callable, optional*
 A function/transform that takes a numpy array of size (1,) and returns a transformed version of the target.

 - **dataset_transform**: *callable, optional*
 A function/transform that takes a dataset (ie. a task), and returns a transformed version of it. E.g. `torchmeta.transforms.ClassSplitter()`.

!!! note "Notes"
    The tasks are created randomly as the sum of two sinusoid functions, with a frequency ratio of 2. The amplitudes vary within [5.0, 7.0], the phases within [0, 2 * pi], and the inputs are sampled according to N(mu_x, 1), with mu_x varying in [-4.0, 4.0]. Due to the way PyTorch handles datasets, the number of tasks to be sampled needs to be fixed ahead of time (with `num_tasks`). This will typically be equal to `meta_batch_size * num_batches`.

!!! attention "References"
    - **[1]** Lacoste A., Oreshkin B., Chung W., Boquet T., Rostamzadeh N., Krueger D. (2018). Uncertainty in Multitask Transfer Learning. In Advances in Neural Information Processing Systems (https://arxiv.org/abs/1806.07528)

## Sinusoid

Simple regression task, based on sinusoids, as introduced in [1].

```python
torchmeta.toy.Sinusoid(num_samples_per_task, num_tasks=1000000, noise_std=None,
    transform=None, target_transform=None, dataset_transform=None)
```

**Parameters**

 - **num_samples_per_task**: *int*
 Number of examples per task.

 - **num_tasks**: *int (default: 1,000,000)*
 Overall number of tasks to sample.

 - **noise_std**: *float, optional*
 Amount of noise to include in the targets for each task. If `None`, then nos noise is included, and the target is a sine function of the input.

 - **transform**: *callable, optional*
 A function/transform that takes a numpy array of size (1,) and returns a transformed version of the input.

 - **target_transform**: *callable, optional*
 A function/transform that takes a numpy array of size (1,) and returns a transformed version of the target.

 - **dataset_transform**: *callable, optional*
 A function/transform that takes a dataset (ie. a task), and returns a transformed version of it. E.g. `torchmeta.transforms.ClassSplitter()`.

!!! note "Notes"
    The tasks are created randomly as random sinusoid function. The amplitude varies within [0.1, 5.0], the phase within [0, pi], and the inputs are sampled uniformly in [-5.0, 5.0]. Due to the way PyTorch handles datasets, the number of tasks to be sampled needs to be fixed ahead of time (with `num_tasks`). This will typically be equal to `meta_batch_size * num_batches`.

!!! attention "References"
    - **[1]** Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. International Conference on Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)

## SinusoidAndLine

Simple multimodal regression task, based on sinusoids and lines, as introduced in [1].

```python
torchmeta.toy.SinusoidAndLine(num_samples_per_task, num_tasks=1000000,
    noise_std=None, transform=None, target_transform=None,
    dataset_transform=None)
```

**Parameters**

 - **num_samples_per_task**: *int*
 Number of examples per task.

 - **num_tasks**: *int (default: 1,000,000)*
 Overall number of tasks to sample.

 - **noise_std**: *float, optional*
 Amount of noise to include in the targets for each task. If `None`, then nos noise is included, and the target is either a sine function, or a linear function of the input.

 - **transform**: *callable, optional*
 A function/transform that takes a numpy array of size (1,) and returns a transformed version of the input.

 - **target_transform**: *callable, optional*
 A function/transform that takes a numpy array of size (1,) and returns a transformed version of the target.

 - **dataset_transform**: *callable, optional*
 A function/transform that takes a dataset (ie. a task), and returns a transformed version of it. E.g. `torchmeta.transforms.ClassSplitter()`.

!!! note "Notes"
    The tasks are created randomly as either random sinusoid functions, or random linear functions. The amplitude of the sinusoids varies within [0.1, 5.0] and the phase within [0, pi]. The slope and intercept of the lines vary in [-3.0, 3.0]. The inputs are sampled uniformly in [-5.0, 5.0]. Due to the way PyTorch handles datasets, the number of tasks to be sampled needs to be fixed ahead of time (with `num_tasks`). This will typically be equal to `meta_batch_size * num_batches`.

!!! attention "References"
    - **[1]** Finn C., Xu K., Levine S. (2018). Probabilistic Model-Agnostic Meta-Learning. In Advances in Neural Information Processing Systems (https://arxiv.org/abs/1806.02817)