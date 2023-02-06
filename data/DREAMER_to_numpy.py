import scipy.io
import numpy as np
import pandas as pd
import os


def main():
    """
    small script to convert DREAMER dataset to numpy array and pandas dataframe
    """
    file_dir = os.path.dirname(__file__)
    dreamer_path = os.path.join(file_dir, 'DREAMER.mat')

    mat = scipy.io.loadmat(dreamer_path)
    data = mat['DREAMER']
    dreamer = data[0, 0]['Data']

    # convert to numpy arrays
    max_length = 50432
    numpy_data_array = np.full((414, 14, max_length), np.NINF)
    numpy_baseline_array = np.zeros((414, 14, 7808))
    numpy_label_array = np.zeros((414, 3))

    # save video lengths
    col = ['sample length']
    df = pd.DataFrame(columns=col)
    position_counter = 0
    # loop over dreamer dataset
    for participant in dreamer[0]:
        # len(participant_data) = 7: Age, Gender, EEG, ECG, Valence, Arousal, Dominance
        participant_data = participant[0, 0]

        for index, data_value in enumerate(participant_data):
            if index == 0 or index == 1 or index == 3:
                continue
            elif index == 2:
                eeg_baseline = data_value[0, 0][0]
                eeg_stimuli = data_value[0, 0][1]

                for baseline_measurement, video_measurement in zip(eeg_baseline, eeg_stimuli):
                    # swap axes
                    baseline_array = np.swapaxes(baseline_measurement[0], 0, 1)
                    stimuli_array = np.swapaxes(video_measurement[0], 0, 1)

                    # write in final array
                    numpy_baseline_array[position_counter] = baseline_array

                    video_length = stimuli_array.shape[1]
                    npad = ((0, 0), (0, max_length - video_length))
                    padded_array = np.pad(stimuli_array, pad_width=npad, mode='constant', constant_values=np.NINF)
                    numpy_data_array[position_counter] = padded_array

                    # store video length in sample size
                    if position_counter < 18:
                        tmp = pd.DataFrame([[video_length]], columns=col)
                        df = pd.concat([df, tmp], ignore_index=True)

                    position_counter += 1

            else:
                idx_to_labelidx = {4: 0, 5: 1, 6: 2}
                numpy_label_array[position_counter - 18:position_counter, idx_to_labelidx[index]] = data_value.flatten()

    # save array
    np.save(os.path.join(file_dir, 'DREAMER_data.npy'), numpy_data_array)
    np.save(os.path.join(file_dir, 'DREAMER_baseline.npy'), numpy_baseline_array)
    np.save(os.path.join(file_dir, 'DREAMER_labels.npy'), numpy_label_array)

    # save video lengths
    df.to_csv(os.path.join(file_dir, 'DREAMER_#samples_per_video.csv'))


if __name__ == "__main__":
    main()
