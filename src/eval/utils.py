import torch


def flatten(nested_list):
    return torch.cat([e for lst in nested_list for e in lst])
