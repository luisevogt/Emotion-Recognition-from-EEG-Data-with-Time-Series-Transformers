import os
import pickle
import numpy as np
import torch

from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


class DEAPDataset(Dataset):
    """
    DEAP dataset.
    """

    sample_freq = 128

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
        self.__sample_freq = DEAPDataset.sample_freq
        self.sample_size = sample_size * self.__sample_freq
        self._trail_num = 40  # 40 videos per participant
        self._sample_num = 60 * self.__sample_freq // self.sample_size
        self._sample_per_part = self._trail_num * self._sample_num

        # save filenames in a list for fast access
        self.filenames = [filename for filename in os.listdir(self.data_dir)
                          if os.path.isfile(os.path.join(self.data_dir, filename))]

        # threshold
        self.__threshold = 4.5

    def get_class_names(self):
        return self.__class_names

    @staticmethod
    def get_channel_grouping():
        group_idx_to_name = {0: 'left frontal region',
                             1: 'right frontal region',
                             2: 'left parietal-temporal-occipital',
                             3: 'right parietal-temporal-occipital'}

        channel_grouping = {0: [0, 1, 2, 3, 4, 5, 6, 23],
                            1: [16, 17, 18, 19, 20, 21, 22, 24],
                            2: [7, 8, 9, 10, 11, 12, 13, 14],
                            3: [15, 25, 26, 27, 28, 29, 30, 31]}
        return group_idx_to_name, channel_grouping

    def __len__(self):
        participant_count = len(self.filenames)

        return participant_count * self._sample_num * self._trail_num

    def __getitem__(self, idx):
        # flatten
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # decompress index
        current_participant = 0
        current_trail = 0
        while idx - current_participant * self._sample_per_part > self._sample_per_part:
            current_participant += 1

        sample_idx = idx - current_participant * self._sample_per_part - current_trail * self._sample_num
        while sample_idx > self._sample_num:
            current_trail += 1
            if current_trail >= self._trail_num:
                current_trail = 0
            sample_idx = idx - current_participant * self._sample_per_part - current_trail * self._sample_num

        # load dataset
        filepath = os.path.join(self.data_dir, self.filenames[current_participant])
        file = pickle.load(open(filepath, 'rb'), encoding='latin1')
        data = file["data"]
        # drop the first three baseline seconds removed
        data = data[:, :, self.__sample_freq * 3:]
        labels = file["labels"]

        # get sample and label
        data_sample = data[current_trail, 0:32, sample_idx:sample_idx + self.sample_size]
        data_sample = np.float32(data_sample)
        label = labels[current_trail][self.__tag_to_idx[self._classification_tag]]

        if label <= self.__threshold:
            label = 0
        elif label > self.__threshold:
            label = 1

        # normalize sample in every channel with min-max normalization
        # scaler = MinMaxScaler()
        # scaler.fit(data_sample)
        # ds_scaled = scaler.transform(data_sample)
        # print(ds_scaled)
        data_sample = np.swapaxes(data_sample, 0, 1)
        return data_sample, label
