import pickle
import json
import random
import numpy as np
import torch
from transformers import set_seed


def set_seed_fn(args):
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "gpu":
        torch.cuda.manual_seed_all(args.seed)


def save_json(outfile_path, result_dict):
    with open(outfile_path, 'w') as f:
        json.dump(result_dict, f)

    print(f'written to {outfile_path}')


def is_correct(model_answer, answer):
    return model_answer == answer

