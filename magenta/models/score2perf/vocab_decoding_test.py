from glob import glob
import os
import sys

import numpy as np
from tqdm import tqdm

from folk_dataset_onehot_encode import encode, decode_seq

sys.path.append("D:\\data\\magenta-1.0.2\\")
sys.path.append("D:\\data\\thesis_model2\\")

original_file = "D:\\data\\folkmagenta_2\\sessiontune1.mid_0.npy"
seq = np.load(original_file).astype('uint8')

one_hot_sequence = encode(seq)

reconstructed = decode_seq(one_hot_sequence)

assert False not in (reconstructed == seq)
print('ALL GOOD')
