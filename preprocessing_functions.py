import numpy as np
import pandas as pd

"""
This function takes ffr data that is current time by trial (meaning trial 1 would be column
1 and the start of each recording would correspond to row 1) and converts it to trial by
time, meaning each row will be 1 trial, and the columns will correspond to the time

Args:
    all_ffr: 2D ffr data you want to transpose; format should be time (row) by trials (column)
    num_trials: the number of trials you have
    all_time: the corresponding time measurements for each individual trial measurement

Returns:
    The transposed ffr (we swap rows and columns)
"""
def transpose_ffr(all_ffr, num_trials, all_time):
    transposed_ffr = np.empty((num_trials, all_time.size))
    for i in range(num_trials):
        transposed_ffr[i,:] = all_ffr[:,i]

    return transposed_ffr

"""
This function takes in a csv file as input and maps each element of your class list to the
the header of the column which contains that same class label in the CSV file. The first row contains the column labels. E.x.

    1,2,3,4
    1,2,3,4         This will map:  1, 5, 9, 13 => 1;     2, 6, 10, 14 =>2;
    5,6,7,8                         3, 7, 11, 15 => 3;    4, 8, 12, 16 => 4.
    9,10,11,12
    13,14,15,16

Args:
    clss_data: The list of class labels you want to map to new class labels
    file_path: the file path of your csv file

Returns:
    A list of new (mapped) class labels
"""
def map_classes(clss_data, file_path):
    df = pd.read_csv(file_path, skipinitialspace=True)

    # Map each element to its column header
    mapped_elements = []

    for element in clss_data:
        for col in df.columns:
            if element in df[col].values:
                mapped_elements.append(col)
                break  # stop after finding the first match

    return mapped_elements

"""
This function will trim the ffr and time file between two specified time stamps. So if your recording
is from -20 to 350 (stimulus occurs at 0) and you ony want to analyze data from 0 to 300,
then trim_ind_1 should be 0, and trim_ind_2 should be 300. They default to min time and
max time respectively

Args:
    ffr: 2D array; format should be trials (row) by time(column)
    all_time: time measurements (in miliseconds) relative to stimulus onset
    trim_ind_1: the time measurement you want to trim everything before it
    trim_ind_2: the time measurement you want to trim everything after it

Returns:
    Tuple of trimmed ffr data, and of new time measurements from trim_ind_1 to trim_ind_2
"""
def trim_ffr(ffr, all_time, trim_index_1=None, trim_index_2=None):
    # Defaults if uninitialized
    if(trim_index_1 is None): trim_index_1 = min(all_time)
    if(trim_index_2 is None): trim_index_2 = max(all_time)

    # Bounds checking
    if(trim_index_1 < trim_index_2): raise ValueError("trim_index_1 must be strictly less than trim_index_2")
    if(trim_index_1 < min(all_time) or trim_index_1 > max(all_time)): raise ValueError("trim_index_1 is out of time range")
    if(trim_index_2 < min(all_time) or trim_index_2 > max(all_time)): raise ValueError("trim_index_2 is out of time range")

    # Trims the ffr and the time
    index1 = 0;
    for i in range(all_time.size):
        if(all_time[i] < trim_index_1): continue
        elif(index1 == 0): index1 = i
        elif(all_time[i] >= trim_index_2):
            ffr = ffr[:, index1:i]
            time = all_time[index1:i]
            break

    return ffr, time

"""
This function will take ffr data and randomly sub-average trials based on the specified
sub-average size and provided class labels. If you intend to map class labels to some
some other class label, it is recommended you do so AFTER sub-averaging.

Args:
    ffr_trials: format should be trial (row) by time (column)
    ffr_trial_clss: list of trial classes for each trial
    sub_average_size: the number of trials you would like averaged together (e.x. input = 100, sa_size = 2,
                      output = 50)
    subject_label: the numeric label of the subject (i.e. subject (1, 2, ..., n)); default is 1 for single subject models

Returns:
    Tuple of sub-averaged trials, their respective class labels, the index of the first data elemet included in the average (i.e. averaging data at indeces 1 though 5), and a list of subject labels for each trial.
"""
def sub_average_data(ffr_trials, ffr_trial_clss, sub_average_size, subject_label=1):
    num_classes = np.unique(ffr_trial_clss).size

    clss = min(np.unique(ffr_trial_clss))
    temp_trials = class_data = [[] for _ in range(num_classes)]
    temp_clss = [[] for _ in range(num_classes)]
    indeces = [[] for _ in range(num_classes)]

    prev = 0
    for i in range(ffr_trials.shape[0]):
        if ffr_trial_clss[i] != clss:
            temp_trials[clss - 1] = ffr_trials[prev:i]
            temp_clss[clss - 1] = ffr_trial_clss[prev:i]
            prev = i
            clss += 1
        indeces[clss - 1].append(i)

    # Handles the final case
    temp_trials[clss - 1] = ffr_trials[prev:]
    temp_clss[clss - 1] = ffr_trial_clss[prev:]

    sub_averaged_trials_start_index = []
    sub_averaged_trials_temp = [[] for _ in range(num_classes)]
    sub_averaged_trial_clss_temp = [[] for _ in range(num_classes)]
    for i in range(num_classes):
        temp_element = np.zeros(ffr_trials.shape[1])  # Initialize as NumPy array

        for j, element in enumerate(temp_trials[i]):
            temp_element += element  # Accumulate trial values

            # When sub_average_size trials are accumulated, compute the mean
            if (j + 1) % sub_average_size == 0:
                temp_element = temp_element / sub_average_size  # Compute mean
                sub_averaged_trials_temp[i].append(temp_element)  # Store the averaged trial
                sub_averaged_trial_clss_temp[i].append(i + 1)  # Store the class label

                sub_averaged_trials_start_index.append(indeces[i][j + 1 - sub_average_size])

                # Reset temp_element and count for the next group
                temp_element = np.zeros(ffr_trials.shape[1])

    # This part of the code combines the lists into 2D arrays instead of 3D ones
    sub_averaged_trials = [element for sublist1 in sub_averaged_trials_temp for element in sublist1]
    sub_averaged_trial_clss = [element for sublist1 in sub_averaged_trial_clss_temp for element in sublist1]
    subject_label_list = [subject_label for _ in range(len(sub_average_size))]

    return sub_averaged_trials, sub_averaged_trial_clss, sub_averaged_trials_start_index, subject_label_list

"""
This function takes ffr trials, the respective classes, and the respective indeces (assuming you sub-averaged)
and returns two stratified, shuffled sets for training and testing based on the provided test_percent param.
We keep track of original indeces in the trials and return them along with train and test data.

Args:
    ffr_trials: format should be trials by time
    ffr_trial_clss: labels for individual ffr trials
    test_percent: the percentage of trials you would like to go to testing
    ffr_trial_indeces: the indeces of original ffr trials if shuffled or sub-averaged

Returns:
    Tuple of train data, train clss, train indeces, test data, test clss, and test indeces
"""
def test_split_stratified(ffr_trials, ffr_trial_clss, test_percent, ffr_trial_indeces=None):
    if(ffr_trial_indeces is None): ffr_trial_indeces = range(ffr_trials.shape[0])

    num_classes = np.unique(ffr_trial_clss).size

    # Separate trials into each different class
    clss = min(np.unique(ffr_trial_clss))
    temp_trials = class_data = [[] for _ in range(num_classes)]
    temp_clss = [[] for _ in range(num_classes)]
    temp_indeces = [[] for _ in range(num_classes)]

    prev = 0
    for i in range(ffr_trials.shape[0]):
        if ffr_trial_clss[i] != clss:
            temp_trials[clss - 1] = ffr_trials[prev:i]
            temp_clss[clss - 1] = ffr_trial_clss[prev:i]
            temp_indeces[clss - 1] = ffr_trial_indeces[prev:i]
            prev = i
            clss += 1
    # Handles the final case
    temp_trials[clss - 1] = ffr_trials[prev:]
    temp_clss[clss - 1] = ffr_trial_clss[prev:]
    temp_indeces[clss - 1] = ffr_trial_indeces[prev:]


    train = [[] for _ in range(num_classes)]
    train_clss = [[] for _ in range(num_classes)]
    train_indeces = [[] for _ in range(num_classes)]
    test = [[] for _ in range(num_classes)]
    test_clss = [[] for _ in range(num_classes)]
    test_indeces = [[] for _ in range(num_classes)]


    for i in range(num_classes):
        # Randomly shuffle the trials for class i
        perm = np.random.permutation(len(temp_trials[i]))
        shuffled_trials = np.array(temp_trials[i])[perm]
        shuffled_indeces = np.array(temp_indeces[i])[perm]

        num_trials = len(shuffled_trials[i])
        test_split_index = num_trials - (int)(num_trials * test_percent)  # 20% goes to testing

        # Splits test data
        test[i] = shuffled_trials[i][test_split_index:]
        test_clss[i] = shuffled_trials[i][test_split_index:]
        test_indeces[i] = shuffled_indeces[i][test_split_index:]
        train[i] = shuffled_trials[i][:test_split_index]
        train_clss[i] = shuffled_trials[i][:test_split_index]
        train_indeces[i] = shuffled_indeces[i][:test_split_index]

    # This part of the code combines the lists into 2D arrays instead of 3D ones
    train_trials = [element for sublist1 in train for element in sublist1]
    train_trial_clss = [element for sublist1 in train_clss for element in sublist1]
    train_indeces = [element for sublist1 in test for element in sublist1]
    test_trials = [element for sublist1 in test for element in sublist1]
    test_trial_clss = [element for sublist1 in test_clss for element in sublist1]
    test_indeces = [element for sublist1 in test for element in sublist1]


    return train_trials, train_clss, train_indeces, test_trials, test_clss, test_indeces

"""
Creates a k number of stratified folds for cross-validation.

Args:
    ffr_trials: Trials by time format data
    ffr_trial_clss: Class labels for each trial
    ffr_trial_indeces: Indices for each trial
    num_folds: Number of folds to create

Returns:
    Tuple of folds_trials, folds_clss, and folds_indeces
"""
def k_fold_stratified(ffr_trials, ffr_trial_clss, ffr_trial_indeces, num_folds):

    num_classes = np.unique(ffr_trial_clss).size
    unique_classes = np.unique(ffr_trial_clss)

    # Initialize lists to hold data for each class
    temp_trials = [[] for _ in range(num_classes)]
    temp_clss = [[] for _ in range(num_classes)]
    temp_indeces = [[] for _ in range(num_classes)]

    # Separate trials by class
    for i, cls in enumerate(unique_classes):
        class_mask = ffr_trial_clss == cls
        temp_trials[i] = ffr_trials[class_mask]
        temp_clss[i] = ffr_trial_clss[class_mask]
        temp_indeces[i] = ffr_trial_indeces[class_mask]

    # Initialize fold containers
    folds_trials = [[] for _ in range(num_folds)]
    folds_clss = [[] for _ in range(num_folds)]
    folds_indeces = [[] for _ in range(num_folds)]

    # Distribute each class's data across folds
    for i in range(num_classes):
        # Randomly shuffle the trials for class i
        num_samples = len(temp_trials[i])
        perm = np.random.permutation(num_samples)
        shuffled_trials = np.array(temp_trials[i])[perm]
        shuffled_clss = np.array(temp_clss[i])[perm]
        shuffled_indeces = np.array(temp_indeces[i])[perm]

        # Calculate base number of samples per fold
        samples_per_fold = num_samples // num_folds
        extras = num_samples % num_folds  # Handle remaining samples

        start_idx = 0
        for j in range(num_folds):
            # Add an extra sample to early folds if we have remainders
            end_idx = start_idx + samples_per_fold + (1 if j < extras else 0)

            fold_trials = shuffled_trials[start_idx:end_idx]
            fold_clss = shuffled_clss[start_idx:end_idx]
            fold_indeces = shuffled_indeces[start_idx:end_idx]

            # Append to appropriate folds
            if len(folds_trials[j]) == 0:
                folds_trials[j] = fold_trials
                folds_clss[j] = fold_clss
                folds_indeces[j] = fold_indeces
            else:
                folds_trials[j] = np.vstack((folds_trials[j], fold_trials))
                folds_clss[j] = np.concatenate((folds_clss[j], fold_clss))
                folds_indeces[j] = np.concatenate((folds_indeces[j], fold_indeces))

            start_idx = end_idx

    return folds_trials, folds_clss, folds_indeces

""" ----- TODO ----- """
''' Add subject tracking to the above code so we know when each subjects data is being tested.
    Implement K-fold cross-validation for my model. Return more data to the user for analytics '''