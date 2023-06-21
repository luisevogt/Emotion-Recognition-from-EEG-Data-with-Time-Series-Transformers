import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import tqdm

from torch.utils.data import Dataset


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

        data_sample = np.swapaxes(data_sample, 0, 1)
        return data_sample, label


class WESADDataset(Dataset):
    """
        WESAD Dataset for testing purposes.
        """

    sample_freq = 700

    def __init__(self, data_dir, sample_size=6, samples_per_file=500):
        """
            :param data_dir: Directory with the datasets from all participants.
            'a' for arousal, 'v' for valence and 'd' for dominance.
            :param sample_size: The size of the sample in seconds. Default is 10.
            """

        self.__class_names = {
            0: "not defined",
            1: "baseline",
            2: "stress",
            3: "amusement",
            4: "meditation",
            5: "should not be used",
        }

        self.__samples_per_file = samples_per_file

        self.data_dir = data_dir
        self.__sample_freq = WESADDataset.sample_freq
        self.sample_size = sample_size * self.__sample_freq
        self.sample_num = 5223 * self.__sample_freq // self.sample_size

        self.filenames = []

        for p_dir in os.listdir(self.data_dir):
            if os.path.isdir(os.path.join(self.data_dir, p_dir)) and 'size' not in p_dir:
                pkl_dir = os.path.join(self.data_dir, p_dir, p_dir + '.pkl')
                self.filenames.append(pkl_dir)

        self.length = len(self.filenames) * self.sample_num

        self.to_samples(self.__samples_per_file)

    def get_class_names(self):
        return self.__class_names

    def to_samples(self, samples_per_file=500):
        """splits data into samples of given sample size to speed up dataloading. Saves a list of targets."""
        print("Write samples...")
        start_time = time.time()

        new_data_path = os.path.join(self.data_dir, f'samples_size_{self.sample_size // self.__sample_freq}')
        # if samples are already there, update
        if os.path.exists(new_data_path) and len(os.listdir(new_data_path)) != 0:
            self.data_dir = new_data_path
            self.filenames = []
            for p_file in os.listdir(self.data_dir):
                if os.path.isfile(os.path.join(self.data_dir, p_file)) and 'targets' not in p_file:
                    self.filenames.append(os.path.join(self.data_dir, p_file))
            self.targets = os.path.join(self.data_dir, f'targets_wesad_size_{self.sample_size // self.__sample_freq}.pkl')
            print("files already exist.")
            return

        Path(new_data_path).mkdir(parents=True, exist_ok=True)

        samples = []
        targets = []
        counter = samples_per_file
        for filename in self.filenames:
            # read file and get data
            file = pickle.load(open(filename, 'rb'), encoding='latin1')
            chest_data = file['signal']['chest']
            data = list(map(lambda x: torch.from_numpy(x), list(chest_data.values())))
            data = torch.cat(data, dim=1)[:3656100]

            label_array = torch.from_numpy(file["label"])[:3656100]

            for sample_idx in range(self.sample_num):
                # get sample and label
                array_idx = sample_idx * self.sample_size
                data_sample = data[array_idx:array_idx + self.sample_size, :].to(torch.float32)
                sample_label = label_array[array_idx:array_idx + self.sample_size]
                label_dist = torch.bincount(sample_label)
                label = torch.argmax(label_dist).item()

                if label == 5 or label == 6 or label == 7:
                    label = 5

                if len(samples) == samples_per_file:
                    with open(os.path.join(new_data_path, f'{counter}_wesad_size_{self.sample_size // self.__sample_freq}.pkl'),
                              'wb') as s_file:
                        sample_array = torch.stack(samples)
                        pickle.dump(sample_array, s_file)

                    counter += samples_per_file
                    samples = []

                samples.append(data_sample)
                targets.append(label)

        with open(os.path.join(new_data_path, f'{counter}_wesad_size_{self.sample_size // self.__sample_freq}.pkl'), 'wb') as s_file:
            sample_array = torch.stack(samples)
            pickle.dump(sample_array, s_file)

        with open(os.path.join(new_data_path, f'targets_wesad_size_{self.sample_size // self.__sample_freq}.pkl'), 'wb') as t_file:
            pickle.dump(targets, t_file)

        self.data_dir = new_data_path
        self.filenames = []
        for p_file in os.listdir(self.data_dir):
            if os.path.isfile(os.path.join(self.data_dir, p_file)) and 'targets' not in p_file:
                self.filenames.append(os.path.join(self.data_dir, p_file))
        self.targets = os.path.join(self.data_dir, f'targets_wesad_size_{self.sample_size // self.__sample_freq}.pkl')

        end_time = time.time()

        el_time = end_time - start_time
        print(f'Wrote samples in {el_time}')

    @staticmethod
    def get_channel_grouping():
        group_idx_to_name = {0: 'ACG',
                             1: 'ECG',
                             2: 'EDA',
                             3: 'BVP',
                             4: 'EMG',
                             5: 'RESP',
                             6: 'TEMP'}

        channel_grouping = {0: [0, 1, 2, 3, 4],
                            1: [5],
                            2: [6, 7],
                            3: [8],
                            4: [9],
                            5: [10],
                            6: [11, 12]}
        return group_idx_to_name, channel_grouping

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # flatten
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_idx = ((idx // self.__samples_per_file) + 1) * self.__samples_per_file
        sample_idx = idx - (idx // self.__samples_per_file) * self.__samples_per_file

        filepath = next((f for f in self.filenames if os.path.basename(f).startswith(str(file_idx))), None)
        data = pickle.load(open(filepath, 'rb'), encoding='latin1')
        labels = pickle.load(open(self.targets, 'rb'), encoding='latin1')
        data_sample = data[sample_idx]
        label = labels[idx]

        return data_sample, label


if __name__ == "__main__":

    filenames = []
    data_dir = "../datasets/WESAD/WESAD"

    dataset = WESADDataset(data_dir)
    print(dataset.__len__())
    # print(dataset.__getitem__(1007))
    [sample[1] for sample in tqdm.tqdm(dataset)]
