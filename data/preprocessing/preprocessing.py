import argparse
import os
import pickle
from pathlib import Path

import mne
from mne import EpochsArray


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocessing")
    parser.add_argument("--data_path", dest="data_path", help="Filepath to data")
    parser.add_argument("--out", dest="out", help="path where outputs are stored")

    args = parser.parse_args()

    data_path = args.data_path
    out_path = args.out
    sfreq = args.sfreq

    standardize(data_path, out_path)
