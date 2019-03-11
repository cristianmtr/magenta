from glob import glob
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.append("D:\\data\\magenta-1.0.2\\")
sys.path.append("D:\\data\\thesis_model2\\")

DATASET_LOC = os.path.abspath("D:\\data\\folkmagenta_2\\")

NEW_LOCATION = os.path.abspath("D:/data/folkmagenta_3")
if not os.path.exists(NEW_LOCATION):
    os.mkdir(NEW_LOCATION)

VOCAB_PATH = os.path.join(NEW_LOCATION, 'vocab.npy')

EVENTS_PER_BAR = 24
NR_BARS = 8
MAX_SEQ_LEN = EVENTS_PER_BAR * NR_BARS

def decode_seq(sequence):
    vocab = np.load(VOCAB_PATH)
    reconstructed_seq = np.repeat(-1, len(sequence))
    for i, step in enumerate(sequence):
        instruction_index = np.argmax(step)
        instruction = vocab[instruction_index]
        reconstructed_seq[i] = instruction
    return reconstructed_seq


def encode(sequence):
    vocab = np.load(VOCAB_PATH)
    new_seq = np.zeros((MAX_SEQ_LEN, len(vocab)), dtype='uint8')
    for instruction_index in range(len(sequence)):
        instruction = sequence[instruction_index]
        instruction_index_in_vocab = np.where(vocab == instruction)[0][0]
        new_seq[instruction_index, instruction_index_in_vocab] = 1
    return new_seq


if __name__ == "__main__":
    DATASET_FILES = glob(os.path.join(DATASET_LOC, "*.npy"))[:100000]

    data = np.zeros((len(DATASET_FILES), MAX_SEQ_LEN), dtype='uint8')

    for i, f in enumerate(tqdm(DATASET_FILES)):
        seq = np.load(f).astype('uint8')
        data[i] = seq

    vocab = np.unique(data)
    np.save(VOCAB_PATH, vocab)

    for seq_i in tqdm(range(len(data))):
        seq = data[seq_i]
        new_seq = encode(seq)
        seq_path = os.path.join(NEW_LOCATION, 'seq_%s' % seq_i)
        np.save(seq_path, new_seq)
