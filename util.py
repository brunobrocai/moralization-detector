import torch
import torch_xla
import torch_xla.core.xla_model as xm


def get_device():
    # Check if TPU is available
    if 'xla' in str(xm.xla_device()):
        device = xm.xla_device()
        print('TPU detected. Using XLA device:', device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(
            'No TPU detected. There are %d GPU(s) available.' % torch.cuda.device_count()
        )
        print('Using GPU:', torch.cuda.get_device_name(0))
    else:
        print('No TPU or GPU detected, using the CPU instead.')
        device = torch.device("cpu")
    return device
