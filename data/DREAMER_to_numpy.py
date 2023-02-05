import scipy.io
import numpy as np
import pandas as pd


def main():
    """
    small script to convert DREAMER dataset to numpy array and pandas dataframe
    """

    mat = scipy.io.loadmat('DREAMER.mat')
    data = mat['DREAMER']
    dreamer = data[0, 0]['Data']
    print(dreamer.shape)

    # flattened data array
    # 23x18 = 414 samples
    # 14 channels
    # 7808 values
    numpy_data_array = np.zeros((414, 14, 7808))
    # numpy_baseline_array = np.zeros((414, 14, 25472))
    numpy_label_array = np.zeros((414, 3))

    cols = ['Age', 'Gender', 'EEG', 'Valence', 'Arousal', 'Dominance']
    df = pd.DataFrame(columns=cols)

    # loop over dreamer dataset
    position_counter = 0
    for participant in dreamer[0]:
        # len(participant_data) = 7: Age, Gender, EEG, ECG, Valence, Arousal, Dominance
        participant_data = participant[0, 0]

        df_row = []
        for index, data_value in enumerate(participant_data):
            if index == 0 or index == 1:
                df_row.append(data_value[0])
            elif index == 3:
                # drop ECG
                continue
            elif index == 2:
                eeg_stimuli = data_value[0, 0][0]
                eeg_baseline = data_value[0, 0][1]

                tmp = []
                for baseline_measurement, video_measurement in zip(eeg_baseline, eeg_stimuli):
                    # swap axes
                    # baseline_array = np.swapaxes(baseline_measurement[0], 0, 1)
                    stimuli_array = np.swapaxes(video_measurement[0], 0, 1)

                    # write in final array
                    numpy_data_array[position_counter] = stimuli_array
                    # numpy_baseline_array[position_counter] = baseline_array
                    tmp.append(stimuli_array)

                df_row.append(tmp)
            else:
                idx_to_labelidx = {4: 0, 5: 1, 6: 2}
                numpy_label_array[position_counter:position_counter + 18, idx_to_labelidx[index]] = data_value.flatten()
                df_row.append(data_value.flatten())

            position_counter += 1

        # append row
        tmp_df = pd.DataFrame([df_row], columns=cols)
        df = pd.concat([df, tmp_df], ignore_index=True)
        position_counter += 1

    # save array and dataframe
    df.to_csv('DREAMER_dataframe.csv')
    np.save('DREAMER_data.npy', numpy_data_array)
    np.save('DREAMER_labels.npy', numpy_label_array)


if __name__ == "__main__":
    main()
