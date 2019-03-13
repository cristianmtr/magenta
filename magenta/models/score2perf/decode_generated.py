import os
import argparse
from my_encoder import encoder
from folk_dataset_onehot_encode import one_hot_decode
import numpy as np

parser = argparse.ArgumentParser(
        description="decode generated npy to midi")
parser.add_argument('file', metavar='file', type=str,
                    help='file path to npy to load')
args = parser.parse_args()

generated = np.load(args.file)
seq = one_hot_decode(generated)
midi_path = encoder.decode(seq)
print(midi_path)
os.startfile(midi_path)

