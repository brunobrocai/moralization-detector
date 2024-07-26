import torch
import pandas as pd


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


def dimi_from_excel(excel_path, sheet_name):
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
    dimi_set = set(df[0].to_list())
    return dimi_set
