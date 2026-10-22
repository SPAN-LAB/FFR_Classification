from ..eeg_subject import EEGSubject
from ..eeg_trial import EEGTrial
from math import floor
from random import Random

def sds2(trials: list[EEGTrial], num_trials: int) -> list[EEGTrial]:
    """
    Returns REFERENCES to the num_trials sampled EEGTrial instances.
    """
    if num_trials < 1:
        raise ValueError("num_trials must be at least 1")
    if num_trials > len(trials):
        raise ValueError(
            f"Cannot sample {num_trials} trials from a pool of {len(trials)}"
        )

    subject = EEGSubject(trials=trials)
    total_num_trials = len(subject.trials)
    grouped_trials = subject.grouped_trials()
    
    # Determine the number of trials each class needs 
    
    # Keys are any (all possible labels/classes of the subject).
    # Values are 2-element tuples, each of which represents the number of trials
    # this class should have, given the number of trials passed to this func.
    # The second element is the floored result of the first.
    num_trials_per_label = {}
    for label, trials in grouped_trials.items():
        n = len(trials) / total_num_trials * num_trials
        num_trials_per_label[label] = [n, floor(n)]
        
    # Determine the distance between the unfloored and floored values
    num_trials_allocated = 0
    distances = []
    for label, num_trials_tuple in num_trials_per_label.items():
        distances.append([label, num_trials_tuple[0]])
        num_trials_allocated += num_trials_tuple[1]
    distances.sort(key=lambda x: x[1], reverse=False)
    
    # Now, distances contains tuples of the form (Label, fractional_trials_needed)
    # sorted by the second value. So we can allocate remaining trials to 
    # classes whose priority is highest
    
    for distance in distances:
        if num_trials_allocated >= num_trials:
            break
        else:
            num_trials_per_label[distance[0]][1] += 1
            num_trials_allocated += 1
    
    if num_trials_allocated != num_trials:
        raise ValueError("Failed sample exactly the specified number of trials")
    
    # Now, the second element in the tuples (values) of num_trials_per_label 
    # corresponds to the number of trials we're going to use for that class
    
    sampled_trials = []
    for label, trials in grouped_trials.items():
        num_trials_to_take = num_trials_per_label[label][1]
        sampled_trials += grouped_trials[label][:num_trials_to_take]
    
    return sampled_trials


def sample_balanced_trials(
    trials: list[EEGTrial], num_trials: int, *, seed: str
) -> list[EEGTrial]:
    """Sample a reproducible, nearly equal number of trials from every label."""
    if num_trials < 1:
        raise ValueError("num_trials must be at least 1")
    if num_trials > len(trials):
        raise ValueError(f"Cannot sample {num_trials} trials from a pool of {len(trials)}")

    grouped: dict[object, list[EEGTrial]] = {}
    for trial in trials:
        grouped.setdefault(trial.label, []).append(trial)

    labels = list(grouped)
    Random(f"{seed}:label-order").shuffle(labels)
    per_label, remainder = divmod(num_trials, len(labels))
    if per_label == 0:
        raise ValueError(f"Need at least {len(labels)} trials to include every label")

    sampled = []
    for index, label in enumerate(labels):
        count = per_label + (index < remainder)
        if count > len(grouped[label]):
            raise ValueError(
                f"Label {label!r} has only {len(grouped[label])} trials; "
                f"need {count} for a balanced sample of {num_trials}"
            )
        shuffled = list(grouped[label])
        Random(f"{seed}:{label!r}").shuffle(shuffled)
        sampled.extend(shuffled[:count])
    return sampled
