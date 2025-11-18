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

        if int(trial.label) == trial.prediction:
            num_correct += 1

    return num_correct / len(subject.trials)
