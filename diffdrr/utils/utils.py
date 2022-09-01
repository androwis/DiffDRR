import torch


def delete_tensor(tensor):
    del tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
