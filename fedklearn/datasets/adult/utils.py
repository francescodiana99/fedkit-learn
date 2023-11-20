from collections import Counter


def print_class_distribution(targets):
    """
    Print the distribution of classes in the target variable.

    Args:
        - targets: List or array-like, the target variable.

    Returns:
        None
    """
    counter = Counter(targets)
    total_samples = len(targets)

    for class_label, count in counter.items():
        percentage = count / total_samples * 100
        print('Class={}, Count={}, Percentage={:.3f}%'.format(class_label, count, percentage))


def print_task_samples_summary(tasks_dict):
    """
    Print a summary of the number of samples for each task.

    Args:
    - tasks_dict (dict): A dictionary containing samples for each task.
    """
    for task_name in tasks_dict:
        print(f"The number of samples for the task '{task_name}' is {len(tasks_dict[task_name])}")
