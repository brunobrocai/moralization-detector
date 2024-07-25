import torch


def get_device():

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(
            'No TPU detected. There are %d GPU(s) available.' % torch.cuda.device_count()
        )
        print('Using GPU:', torch.cuda.get_device_name(0))
    else:
        print('No TPU or GPU detected, using the CPU instead.')
        device = torch.device("cpu")
    return device
