import glob
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas
import torch
import scipy.io

from torch.utils.data import Dataset
import tqdm
import mne

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


class SEEDDataset(Dataset):
    """
    SEED dataset.
    """

    sample_freq = 200

    def __init__(self, data_dir, sample_size=6, save_per_file=250):
        self.__class_names = {0: 'negative',
                              1: 'neutral',
                              2: 'positive'}

        self.__label_to_class = {
            -1: 0,
            0: 1,
            1: 2
        }

        self.__save_per_file = save_per_file

        self.__trail_lengths = [47001, 46601, 41201, 47601, 37001, 39001, 47401, 43201, 53001, 47401, 47001,
                                46601, 47001, 47601, 41201]

        self.data_dir = data_dir
        self.__sample_freq = SEEDDataset.sample_freq
        self.sample_size = sample_size * self.__sample_freq
        self._trail_num = 15
        self.samples_per_trail = [t_len // self.sample_size for t_len in self.__trail_lengths]
        self._samples_per_file = sum(self.samples_per_trail)

        # save filenames in a list for fast access
        self.filenames = glob.glob(os.path.join(data_dir, '*_*.mat'))

        self.length = len(self.filenames) * self._samples_per_file

        self.to_samples(self.__save_per_file)

    def get_class_names(self):
        return self.__class_names

    def to_samples(self, samples_per_file):
        """
        splits data into samples of given size to speed up dataloading. Also saves a list of targets.
        """

        print("Write samples...")
        start_time = time.time()

        new_data_path = os.path.join(self.data_dir, f'samples_size_{self.sample_size // self.sample_freq}')
        # is samples are already there, update paths
        if os.path.exists(new_data_path) and len(os.listdir(new_data_path)) != 0:
            self.data_dir = new_data_path
            self.filenames = glob.glob(os.path.join(self.data_dir, '*_seed_size_*.pkl'))
            self.targets = os.path.join(new_data_path,
                                        f'targets_seed_size_{self.sample_size // self.__sample_freq}.pkl')

            print("files already exist.")
            return

        Path(new_data_path).mkdir(parents=True, exist_ok=True)

        samples = []
        targets = []
        counter = samples_per_file

        labels = scipy.io.loadmat(os.path.join(self.data_dir, "label.mat"))['label'].tolist()[0]

        for filename in self.filenames:

            # load data
            file = scipy.io.loadmat(filename)
            key = list(file.keys())[3][:-1]
            for trail_idx in range(self._trail_num):
                data = file[key + str(trail_idx + 1)]
                # scale
                #scaler = mne.decoding.Scaler(scalings='mean')
                #data = scaler.fit_transform(data)
                #data = scaler.fit_transform(data.T)[:,:,0].T
                for sample_idx in range(self.samples_per_trail[trail_idx]):
                    # get sample and label
                    array_idx = sample_idx * self.sample_size
                    data_sample = data[:, array_idx:array_idx + self.sample_size]
                    data_sample = np.float32(data_sample)
                    data_sample = np.swapaxes(data_sample, 0, 1)
                    label = self.__label_to_class[labels[trail_idx]]

                    if len(samples) == samples_per_file:
                        with open(os.path.join(new_data_path,
                                               f'{counter}_seed_size_{self.sample_size // self.__sample_freq}.pkl'),
                                  'wb') as s_file:
                            sample_array = np.stack(samples)
                            pickle.dump(sample_array, s_file)
                        counter += samples_per_file
                        samples = []

                    samples.append(data_sample)
                    targets.append(label)

        with open(os.path.join(new_data_path, f'{counter}_seed_size_{self.sample_size // self.__sample_freq}.pkl'),
                  'wb') as s_file:
            sample_array = np.stack(samples)
            pickle.dump(sample_array, s_file)

        with open(os.path.join(new_data_path, f'targets_seed_size_{self.sample_size // self.__sample_freq}.pkl'),
                  'wb') as t_file:
            pickle.dump(targets, t_file)

        self.data_dir = new_data_path
        self.filenames = glob.glob(os.path.join(self.data_dir, '*_seed_size_*.pkl'))
        self.targets = os.path.join(new_data_path, f'targets_seed_size_{self.sample_size // self.__sample_freq}.pkl')

        end_time = time.time()

        el_time = end_time - start_time
        print(f'Wrote samples in {el_time}')

    @staticmethod
    def get_channel_grouping():
        group_idx_to_name = {0: 'left frontal region',
                             1: 'right frontal region',
                             2: 'left parietal-temporal-occipital',
                             3: 'right parietal-temporal-occipital',
                             4: 'center'}

        channel_grouping = {0: [0,3, 5, 6, 7, 8, 14, 15, 16, 17],
                            1: [2, 4, 10, 11, 12, 13, 19, 20, 21, 22],
                            2: [32, 33, 34, 35, 41, 42, 43, 44, 50, 51, 52, 57, 58],
                            3: [37, 38, 39, 40, 46, 47, 48, 49, 54, 55, 56, 60, 61],
                            4: [1, 9, 18, 23, 24, 25, 26, 27, 28, 29, 30, 31, 36, 45, 53, 59]}

        return group_idx_to_name, channel_grouping

    def __len__(self):

        return self.length

    def __getitem__(self, idx):

        # flatten
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_idx = ((idx // self.__save_per_file) + 1) * self.__save_per_file
        sample_idx = idx - (idx // self.__save_per_file) * self.__save_per_file

        filepath = next((f for f in self.filenames if os.path.basename(f).startswith(str(file_idx))), None)
        data = pickle.load(open(filepath, 'rb'), encoding='latin1')
        labels = pickle.load(open(self.targets, 'rb'), encoding='latin1')
        data_sample = data[sample_idx]
        label = labels[idx]
        # current_file = 0
        # current_trial = 0
        #
        # while idx - current_file * self._samples_per_file >= self._samples_per_file:
        #     current_file += 1
        #
        # file_idx = idx - current_file * self._samples_per_file
        # sample_idx = file_idx - sum(self.samples_per_trail[:current_trial])
        # while sample_idx >= 0:
        #     current_trial += 1
        #     sample_idx = file_idx - sum(self.samples_per_trail[:current_trial])
        # if current_trial != 0:
        #     current_trial -= 1
        # sample_idx = self.samples_per_trail[current_trial] + sample_idx  # sample_idx is negative
        #
        # sample_idx = sample_idx * self.sample_size
        #
        # # load data
        # file = scipy.io.loadmat(self.filenames[current_file])
        # key = list(file.keys())[3][:-1]
        # data = file[key + str(current_trial + 1)]
        #
        # # get sample and label
        # data_sample = data[:, sample_idx:sample_idx + self.sample_size]
        # data_sample = np.float32(data_sample)
        # label = self.__label_to_class[self.labels[current_trial]]
        #
        # data_sample = np.swapaxes(data_sample, 0, 1)

        return data_sample, label


if __name__ == '__main__':
    path = '../datasets/SEED_EEG/Preprocessed_EEG'

    dataset = SEEDDataset(path)

    print(dataset.__len__())

    [sample[1] for sample in tqdm.tqdm(dataset)]
