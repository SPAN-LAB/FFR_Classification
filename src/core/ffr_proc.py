from .eeg_subject import EEGSubject


def get_accuracy(subject: EEGSubject, enforce_saturation: bool = False) -> float:
    """
    Calculates the accuracy of predictions made on the provided subject.

    :param subject: the subject to determine the accuracy of predictions on
    :returns: a float between 0 and 1 representing the accuracy of the predictions
    """
    num_correct = 0
    total_count = 0
    for trial in subject.trials:
        # Enforce that all trials have predicitions iff this condition is enforced
        if trial.prediction is None:
            if enforce_saturation:
                raise ValueError("Expected prediction not found.")
        else:
            total_count += 1

        if trial.label == trial.prediction:
            num_correct += 1
    if total_count == 0:
        return float("nan")
    return num_correct / total_count

def get_per_label_accuracy(subject: EEGSubject, enforce_saturation: bool = False) -> dict[any, float]:
    """
    Initialize the results dictionary, where each value is a 2-element tupple containing the number
    of correctly classified trials and the total number of trials for that label respectively
    """
    results = {}
    for label in subject.labels_map.keys():
        results[label] = [0, 0]

    for trial in subject.trials:
        if trial.prediction is None:
            if enforce_saturation:
                raise ValueError("Expected prediction not found.")
        else:
            results[trial.label][1] += 1
        
        if trial.label == trial.prediction:
            results[trial.label][0] += 1

    output = {}
    for key, value in results.items():
        if value[1] != 0:
            output[key] = value[0] / value[1]
        else:
            output[key] = float("nan")
    return output
