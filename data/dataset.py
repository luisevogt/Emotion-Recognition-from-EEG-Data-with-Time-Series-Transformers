import os
import pickle

import torch
from torch.utils.data import Dataset


class DEAPDataset(Dataset):
    """
    DEAP dataset.
    """

    def __init__(self, data_dir, classification_tag, sample_size=10):
        """
        :param data_dir: Directory with the datasets from all participants.
        :param classification_tag: A character that indicates which label should be used from data. Valid tags are
        'a' for arousal, 'v' for valence and 'd' for dominance.
        :param sample_size: The size of the sample in seconds. Default is 10.
        """

        # set classification tag
        assert classification_tag.lower() in ['a', 'v', 'd'], "Please provide a valid classification tag. " \
                                                              "Valid tags are: a, v, d for arousal, valence and " \
                                                              "dominance. "
        self._classification_tag = classification_tag

        if classification_tag.lower() == 'a':
            self.__class_names = {0: "low arousal",
                                  1: "high arousal"}
        elif classification_tag.lower() == 'v':
            self.__class_names = {0: "low valence",
                                  1: "high valence"}
        elif classification_tag.lower() == 'd':
            self.__class_names = {0: "low dominance",
                                  1: "high dominance"}
        else:
            raise ValueError("Please provide a valid classification tag. Valid tags are: a, v, d for arousal, "
                             "valence and dominance.")

        # class indexes in DEAP dataset
        self.__tag_to_idx = {
            'v': 0,
            'a': 1,
            'd': 2
        }

        # the preprocessed DEAP datasets consists of 60 second trails
        assert 60 % sample_size == 0, "The sample size should be a factor of 60."

        self.data_dir = data_dir

        self.__sample_freq = 128
        self.sample_size = sample_size * self.__sample_freq
        self._trail_num = 40  # 40 videos per participant
        self._sample_num = 60 * self.__sample_freq // self.sample_size

        # save filenames in a list for fast access
        self.filenames = [filename for filename in os.listdir(self.data_dir)
                          if os.path.isfile(os.path.join(self.data_dir, filename))]

        # keep track of the current participant and current trail
        self.current_participant = 0
        self.current_trail = 0

        # threshold
        self.__threshold = 4.5

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

        # load dataset
        filepath = os.path.join(self.data_dir, self.filenames[self.current_participant])
        file = pickle.load(open(filepath, 'rb'), encoding='latin1')
        data = file["data"]
        labels = file["labels"]

        # get sample
        data_sample = data[self.current_trail, 0:32, sample_idx:sample_idx + self.sample_size]
        label = labels[self.current_trail][self.__tag_to_idx[self._classification_tag]]

        if label < self.__threshold:
            label = [0]
        elif label > self.__threshold:
            label = [1]

        sample = {'datasets': data_sample, 'labels': label}

        return sample

