import glob
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import Dataset
from collections import Counter


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
                          if os.path.isfile(os.path.join(self.data_dir, filename)) and 'targets' not in filename]

        # threshold
        self.__threshold = 4.5

        # write targets
        self.get_targets()

    def get_class_names(self):
        return self.__class_names

    def get_targets(self):
        """Saves is list of targets."""
        print("Get targets")
        start_time = time.time()

        target_path = os.path.join(self.data_dir, f'targets_deap_size_{self.sample_size // self.__sample_freq}.pkl')

        # if targets are already there, update targets field
        if os.path.exists(target_path):
            self.targets = target_path
            print("targets already exist.")
            return

        targets = []
        for filename in self.filenames:
            # load file
            filepath = os.path.join(self.data_dir, filename)
            file = pickle.load(open(filepath, 'rb'), encoding='latin1')
            labels = file["labels"]
            for trail_idx in range(self._trail_num):
                # get label
                label = labels[trail_idx][self.__tag_to_idx[self._classification_tag]]

                if label <= self.__threshold:
                    label = 0
                elif label > self.__threshold:
                    label = 1

                targets.append(label)

        # save targets and update target field
        with open(target_path, 'wb') as t_file:
            pickle.dump(targets, t_file)

        self.targets = target_path

        end_time = time.time()
        el_time = end_time - start_time
        print(f'Wrote targets in {el_time}.')

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
        while idx - current_participant * self._sample_per_part >= self._sample_per_part:
            current_participant += 1

        sample_idx = idx - current_participant * self._sample_per_part - current_trail * self._sample_num
        while sample_idx >= self._sample_num:
            current_trail += 1
            if current_trail >= self._trail_num:
                current_trail = 0
            sample_idx = idx - current_participant * self._sample_per_part - current_trail * self._sample_num

        sample_idx = sample_idx * self.sample_size

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

        data_sample = np.swapaxes(data_sample, 0, 1)
        return data_sample, label


class DreamerDataset(Dataset):
    """
    DREAMER Dataset.
    """

    sample_freq = 128

    def __init__(self, data_dir, classification_tag, sample_size=10, stdize=True):
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

        # class indexes in DREAMER dataset
        self.__tag_to_idx = {
            'v': 0,
            'a': 1,
            'd': 2
        }

        assert 50 % sample_size == 0, "The sample size should be a factor of 50."

        self.data_dir = data_dir
        self.__sample_freq = DreamerDataset.sample_freq
        self.sample_size = sample_size * self.__sample_freq
        self._trail_num = 18  # 18 videos per participant
        self._sample_num = 50 * self.__sample_freq // self.sample_size
        self._sample_per_part = self._trail_num * self._sample_num

        # save filename
        self.filenames = glob.glob(os.path.join(self.data_dir, 'subj*.pkl'))

        self.length = len(self.filenames) * self._sample_num * self._trail_num

        # threshold
        self.__threshold = 2.5

        # write targets
        self.oversampling()

    def get_class_names(self):
        return self.__class_names

    def oversampling(self):
        print("Oversample ...")
        start_time = time.time()

        new_path = os.path.join(self.data_dir, f'oversampled_sample_size_{self.sample_size // self.__sample_freq}_{self._classification_tag}')

        Path(new_path).mkdir(parents=True, exist_ok=True)

        samples = []
        targets = []

        for filepath in self.filenames:
            file = pickle.load(open(filepath, 'rb'), encoding='latin1')
            data = file["data"]
            labels = file["labels"]
            for trail_idx in range(self._trail_num):
                label = labels[trail_idx][self.__tag_to_idx[self._classification_tag]]
                if label <= self.__threshold:
                    label = 0
                elif label > self.__threshold:
                    label = 1
                for sample_idx in range(self._sample_num):
                    array_idx = sample_idx * self.sample_size
                    data_sample = data[trail_idx, :, array_idx:array_idx + self.sample_size]
                    data_sample = np.float32(data_sample)
                    samples.append(data_sample)
                    targets.append(label)
                    if label == 0:
                        samples.append(data_sample)
                        targets.append(label)

        samples_res = np.stack(samples, axis=0)
        targets_res = np.array(targets)

        self.length = len(samples_res)
        print(Counter(targets_res))

        self.save_per_file = len(samples_res) // 8
        for i in range(0, len(samples_res), self.save_per_file):
            split = samples_res[i:i + self.save_per_file]
            with open(os.path.join(new_path, f'{i}_samples_dreamer.pkl'), 'wb') as s_file:
                pickle.dump(split, s_file)
        with open(os.path.join(new_path, 'targets_dreamer.pkl'), 'wb') as t_file:
            pickle.dump(targets_res, t_file)

        self.data_dir = new_path
        self.filenames = glob.glob(os.path.join(self.data_dir, '*_samples_dreamer.pkl'))
        self.targets = os.path.join(new_path, 'targets_dreamer.pkl')

        end_time = time.time()

        el_time = end_time - start_time
        print(f'Oversampled in {el_time}')

    def get_targets(self):
        """Saves is list of targets."""
        print("Get targets")
        start_time = time.time()

        target_path = os.path.join(self.data_dir,
                                   f'targets_dreamer_{self._classification_tag}_size_{self.sample_size // self.__sample_freq}.pkl')

        # if targets are already there, update targets field
        if os.path.exists(target_path):
            self.targets = target_path
            print("targets already exist.")
            return

        targets = []
        for filename in self.filenames:
            # load file
            file = pickle.load(open(filename, 'rb'), encoding='latin1')
            labels = file["labels"]
            for trail_idx in range(self._trail_num):
                # get label
                label = labels[trail_idx][self.__tag_to_idx[self._classification_tag]]

                if label <= self.__threshold:
                    label = 0
                elif label > self.__threshold:
                    label = 1

                targets.extend([label] * self._sample_num)

        # save targets and update target field
        with open(target_path, 'wb') as t_file:
            pickle.dump(targets, t_file)

        self.targets = target_path

        end_time = time.time()
        el_time = end_time - start_time
        print(f'Wrote targets in {el_time}.')

    @staticmethod
    def get_channel_grouping():
        group_idx_to_name = {0: 'left frontal region',
                             1: 'right frontal region',
                             2: 'left parietal-temporal-occipital',
                             3: 'right parietal-temporal-occipital'}

        channel_grouping = {0: [0, 1, 2, 3],
                            1: [10, 11, 12, 13],
                            2: [4, 5, 6],
                            3: [7, 8, 9]}
        return group_idx_to_name, channel_grouping

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # flatten
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # decompress index
        file_idx = ((idx // self.save_per_file)) * self.save_per_file
        sample_idx = idx - (idx // self.save_per_file) * self.save_per_file

        # load dataset
        filepath = next((f for f in self.filenames if os.path.basename(f).startswith(str(file_idx))), None)
        data = pickle.load(open(filepath, 'rb'), encoding='latin1')
        labels = pickle.load(open(self.targets, 'rb'), encoding='latin1')
        data_sample = data[sample_idx]
        label = labels[idx]

        data_sample = np.swapaxes(data_sample, 0, 1)

        return data_sample, label


