import numpy as np
from typing import Dict, Tuple, List

train_pos_path = 'data/train_pos.npy'
train_neg_path = 'data/train_neg.npy'
kg_path = 'data/kg.txt'


def load_kg_data(kg_file: str) -> List[str]:
    with open(kg_file, 'r') as file:
        kg_lines = file.readlines()
    return kg_lines


def load_train_data(train_pos_path: str,train_neg_path: str):
    return np.load(train_pos_path), np.load(train_neg_path)
