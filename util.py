import torch


def get_device():

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(
            'No TPU detected. '
            f'There are {torch.cuda.device_count()} GPU(s) available.'
        )
        print('Using GPU:', torch.cuda.get_device_name(0))
    else:
        print('No TPU or GPU detected, using the CPU instead.')
        device = torch.device("cpu")
    return device


def results_to_dict(results):
    return [
        result.to_dict() for result in results
    ]
