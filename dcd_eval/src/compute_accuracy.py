import json
import argparse
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('--data_path', type=str, default='',
                        help='')
    args = parser.parse_args()
    return args


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    args = parse_arguments()
    data = np.array(load_json(args.data_path)['is_correct'])
    filtered_true = data == True
    print((len(data[filtered_true]) / len(data)) * 100)