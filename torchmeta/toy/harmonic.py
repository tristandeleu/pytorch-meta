import numpy as np

from torchmeta.utils.data import Task, MetaDataset


class Harmonic(MetaDataset):
    """
    Simple regression task, based on the sum of two sine waves, as introduced
    in [1].

    Parameters
    ----------
    num_samples_per_task : int
        Number of examples per task.

    num_tasks : int (default: 5,000)
        Overall number of tasks to sample.

    noise_std : float, optional
        Amount of noise to include in the targets for each task. If `None`, then
        nos noise is included, and the target is the sum of 2 sine functions of
        the input.

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
    The tasks are created randomly as the sum of two sinusoid functions, with
    a frequency ratio of 2. The amplitudes vary within [5.0, 7.0], the phases
    within [0, 2 * pi], and the inputs are sampled according to N(mu_x, 1), with
    mu_x varying in [-4.0, 4.0]. Due to the way PyTorch handles datasets, the
    number of tasks to be sampled needs to be fixed ahead of time (with
    `num_tasks`). This will typically be equal to `meta_batch_size * num_batches`.

    References
    ----------
    .. [1] Lacoste A., Oreshkin B., Chung W., Boquet T., Rostamzadeh N.,
           Krueger D. (2018). Uncertainty in Multitask Transfer Learning. In
           Advances in Neural Information Processing Systems (https://arxiv.org/abs/1806.07528)
    """
    def __init__(self, num_samples_per_task, num_tasks=5000,
                 noise_std=None, transform=None, target_transform=None,
                 dataset_transform=None):
        super(Harmonic, self).__init__(meta_split='train',
            target_transform=target_transform, dataset_transform=dataset_transform)
        self.num_samples_per_task = num_samples_per_task
        self.num_tasks = num_tasks
        self.noise_std = noise_std
        self.transform = transform

        self._domain_range = np.array([-4.0, 4.0])
        self._frequency_range = np.array([5.0, 7.0])
        self._phase_range = np.array([0, 2 * np.pi])

        self._domains = None
        self._frequencies = None
        self._phases = None
        self._amplitudes = None

    @property
    def domains(self):
        if self._domains is None:
            self._domains = self.np_random.uniform(self._domain_range[0],
                self._domain_range[1], size=self.num_tasks)
        return self._domains

    @property
    def frequencies(self):
        if self._frequencies is None:
            self._frequencies = self.np_random.uniform(self._frequency_range[0],
                self._frequency_range[1], size=self.num_tasks)
        return self._frequencies

    @property
    def phases(self):
        if self._phases is None:
            self._phases = self.np_random.uniform(self._phase_range[0],
                self._phase_range[1], size=(self.num_tasks, 2))
        return self._phases

    @property
    def amplitudes(self):
        if self._amplitudes is None:
            self._amplitudes = self.np_random.randn(self.num_tasks, 2)
        return self._amplitudes

    def __len__(self):
        return self.num_tasks

    def __getitem__(self, index):
        domain = self.domains[index]
        frequency = self.frequencies[index]
        phases = self.phases[index]
        amplitudes = self.amplitudes[index]

        task = HarmonicTask(index, domain, frequency, phases, amplitudes,
            self.noise_std, self.num_samples_per_task, self.transform,
            self.target_transform, np_random=self.np_random)

        if self.dataset_transform is not None:
            task = self.dataset_transform(task)

        return task


class HarmonicTask(Task):
    def __init__(self, index, domain, frequency, phases, amplitudes,
                 noise_std, num_samples, transform=None,
                 target_transform=None, np_random=None):
        super(HarmonicTask, self).__init__(index, None) # Regression task
        self.domain = domain
        self.frequency = frequency
        self.phases = phases
        self.amplitudes = amplitudes
        self.noise_std = noise_std
        self.num_samples = num_samples

        self.transform = transform
        self.target_transform = target_transform

        if np_random is None:
            np_random = np.random.RandomState(None)

        a_1, a_2 = self.amplitudes
        b_1, b_2 = self.phases

        self._inputs = self.domain + np_random.randn(num_samples, 1)
        self._targets = (a_1 * np.sin(frequency * self._inputs + b_1)
            + a_2 * np.sin(2 * frequency * self._inputs + b_2))
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
