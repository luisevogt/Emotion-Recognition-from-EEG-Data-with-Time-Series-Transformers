import os
import pickle

import torch
from torch.utils.data import Dataset


class DEAPDataset(Dataset):
    """
    DEAP dataset.
    """

    def __init__(self, data_dir, sample_size=10):
        """

        :param data_dir: Directory with the datasets from all participants.
        :param sample_size: The size of the sample in seconds. Default is 10.
        """

        # TODO: label in here, if arousal valence or dominance and select the right one later in code

        # the preprocessed DEAP datasets consists of 60 second trails
        assert 60 % sample_size == 0, "The sample size should be a factor of 60."

        self.data_dir = data_dir

        self.sample_size = sample_size * 128  # 128 is the sample frequency
        self._trail_num = 40  # 40 videos per participant
        self._sample_num = 60 * 128 // self.sample_size

        # save filenames in a list for fast access
        self.filenames = [filename for filename in os.listdir(self.data_dir)
                          if os.path.isfile(os.path.join(self.data_dir, filename))]

        # keep track of the current participant and current trail
        self.current_participant = 0
        self.current_trail = 0

    def __len__(self):
        participant_count = len(self.filenames)

        return participant_count * self._sample_num * self._trail_num

    def __getitem__(self, idx):
        # flatten
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # decompress index
        sample_per_part = self._trail_num * self._sample_num
        while idx - self.current_participant * sample_per_part > sample_per_part:
            self.current_participant += 1
            self.current_trail = 0

        sample_idx = idx - self.current_participant * sample_per_part - self.current_trail * self._sample_num
        while sample_idx > self._sample_num:
            self.current_trail += 1
            if self.current_trail >= self._trail_num:
                self.current_trail = 0
            sample_idx = idx - self.current_participant * sample_per_part - self.current_trail * self._sample_num

        # load datasets
        filepath = os.path.join(self.data_dir, self.filenames[self.current_participant])
        file = pickle.load(open(filepath, 'rb'), encoding='latin1')
        data = file["datasets"]
        labels = file["labels"]

        # get sample
        data_sample = data[self.current_trail, :, sample_idx:sample_idx + self.sample_size]
        label = labels[self.current_trail]
        sample = {'datasets': data_sample, 'labels': label}

        return sample

