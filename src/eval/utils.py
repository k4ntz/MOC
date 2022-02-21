import torch


def flatten(nested_list):
    print(nested_list[0][0].shape)
    return torch.cat([e for lst in nested_list for e in lst])
