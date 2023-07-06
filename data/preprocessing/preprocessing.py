import argparse
import os
import pickle
from pathlib import Path

import mne
import numpy as np
from scipy.io import loadmat


def standardize(data_path, output_path):
    # get filenames
    filenames = [filename for filename in os.listdir(data_path)
                 if os.path.isfile(os.path.join(data_path, filename))]

    # create output folder if not existent
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # read data
    for filename in filenames:
        filepath = os.path.join(data_path, filename)
        file = pickle.load(open(filepath, 'rb'), encoding='latin1')

        data = file["data"]
        labels = file["labels"]

        # scale
        scaler = mne.decoding.Scaler(scalings='mean')
        scaled_data = scaler.fit_transform(data)

        pickle.dump({'data': scaled_data, 'labels': labels}, open(os.path.join(output_path, filename), 'wb'))


def split_dreamer(data_path, crop=50, std=True):
    # open dreamer
    path = 'subj_files_no_std'
    mat = loadmat(data_path + 'DREAMER.mat')
    fs_eeg = mat["DREAMER"]['EEG_SamplingRate'][0, 0][0][0]
    subj_num = 23

    Path(os.path.join(data_path, path)).mkdir(parents=True, exist_ok=True)

    for subj_idx in range(subj_num):
        subject_data = mat["DREAMER"]["Data"][0, 0][0, subj_idx]
        eeg_stimuli = subject_data["EEG"][0, 0]["stimuli"][0, 0]
        trail_num = eeg_stimuli.shape[0]
        trail_data = []
        for trail_idx in range(trail_num):
            # get trail data
            trail_d = eeg_stimuli[trail_idx, 0]
            # crop
            trail_d = np.flip(np.flip(trail_d, axis=0)[:fs_eeg * crop, :])
            # append
            trail_data.append(trail_d)

        data = np.stack(trail_data, axis=0)
        data = np.swapaxes(data, 1, 2)
        if std:
            # scale
            scaler = mne.decoding.Scaler(scalings='mean')
            data = scaler.fit_transform(data)

        # get labels
        labels = [subject_data["ScoreValence"][0, 0], subject_data["ScoreArousal"][0, 0],
                  subject_data["ScoreDominance"][0, 0]]
        labels = np.concatenate((labels[0], labels[1], labels[2]), axis=1)

        data_dict = {'data': data, 'labels':labels}

        # write file
        with open(os.path.join(data_path, path, f'subj{subj_idx}.pkl'), 'wb') as file:
            pickle.dump(data_dict, file)


if __name__ == '__main__':
    # change file path when changing std
    split_dreamer('../../datasets/DREAMER/', std=False)
