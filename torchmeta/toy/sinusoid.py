import numpy as np

from torchmeta.utils.data import Task, MetaDataset


class Sinusoid(MetaDataset):
    """
    Simple regression task, based on sinusoids, as introduced in [1].

    Parameters
    ----------
    num_samples_per_task : int
        Number of examples per task.

    num_tasks : int (default: 1,000,000)
        Overall number of tasks to sample.

    noise_std : float, optional
        Amount of noise to include in the targets for each task. If `None`, then
        nos noise is included, and the target is a sine function of the input.

    transform : callable, optional
        A function/transform that takes a numpy array of size (1,) and returns a
        transformed version of the input.

    target_transform : callable, optional
        A function/transform that takes a numpy array of size (1,) and returns a
        transformed version of the target.

    dataset_transform : callable, optional
        A function/transform that takes a dataset (ie. a task), and returns a 
        transformed version of it. E.g. `torchmeta.transforms.ClassSplitter()`.

    Notes
    -----
    The tasks are created randomly as random sinusoid function. The amplitude
    varies within [0.1, 5.0], the phase within [0, pi], and the inputs are
    sampled uniformly in [-5.0, 5.0]. Due to the way PyTorch handles datasets,
    the number of tasks to be sampled needs to be fixed ahead of time (with
    `num_tasks`). This will typically be equal to `meta_batch_size * num_batches`.

    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    """
    def __init__(self, num_samples_per_task, num_tasks=1000000,
                 noise_std=None, transform=None, target_transform=None,
                 dataset_transform=None):
        super(Sinusoid, self).__init__(meta_split='train',
            target_transform=target_transform, dataset_transform=dataset_transform)
        self.num_samples_per_task = num_samples_per_task
        self.num_tasks = num_tasks
        self.noise_std = noise_std
        self.transform = transform

        self._input_range = np.array([-5.0, 5.0])
        self._amplitude_range = np.array([0.1, 5.0])
        self._phase_range = np.array([0, np.pi])

        self._amplitudes = None
        self._phases = None

    @property
    def amplitudes(self):
        if self._amplitudes is None:
            self._amplitudes = self.np_random.uniform(self._amplitude_range[0],
                self._amplitude_range[1], size=self.num_tasks)
        return self._amplitudes

    @property
    def phases(self):
        if self._phases is None:
            self._phases = self.np_random.uniform(self._phase_range[0],
                self._phase_range[1], size=self.num_tasks)
        return self._phases

    def __len__(self):
        return self.num_tasks

    def __getitem__(self, index):
        amplitude, phase = self.amplitudes[index], self.phases[index]
        task = SinusoidTask(index, amplitude, phase, self._input_range,
            self.noise_std, self.num_samples_per_task, self.transform,
            self.target_transform, np_random=self.np_random)

        if self.dataset_transform is not None:
            task = self.dataset_transform(task)

        return task


class SinusoidTask(Task):
    def __init__(self, index, amplitude, phase, input_range, noise_std,
                 num_samples, transform=None, target_transform=None,
                 np_random=None):
        super(SinusoidTask, self).__init__(index, None) # Regression task
        self.amplitude = amplitude
        self.phase = phase
        self.input_range = input_range
        self.num_samples = num_samples
        self.noise_std = noise_std

        self.transform = transform
        self.target_transform = target_transform

        if np_random is None:
            np_random = np.random.RandomState(None)

        self._inputs = np_random.uniform(input_range[0], input_range[1],
            size=(num_samples, 1))
        self._targets = amplitude * np.sin(self._inputs - phase)
        if (noise_std is not None) and (noise_std > 0.):
            self._targets += noise_std * np_random.randn(num_samples, 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        input, target = self._inputs[index], self._targets[index]

        if self.transform is not None:
            input = self.transform(input)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (input, target)
