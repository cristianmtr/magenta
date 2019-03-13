from glob import glob
import os
import sys

# make sure we use the same settings
from encoders import *
from my_encoder import encoder as folk_encoder
import numpy as np
from tqdm import tqdm
from utils import compute_sequences_available

sys.path.append("D:\\data\\magenta-1.0.2\\")
sys.path.append("D:\\data\\thesis_model2\\")


EVENTS_PER_BAR = 24
NR_BARS = 8
MAX_SEQ_LEN = EVENTS_PER_BAR * NR_BARS


DATASET_LOC = os.path.abspath("D:\\data\\folkdataset\\")
DATASET_GLOB = os.path.join(DATASET_LOC, "*.mid")
DATASET_FILES = glob(DATASET_GLOB)
LOCATION_1 = os.path.abspath("D:\\data\\folkmagenta_2\\")

if not os.path.exists(LOCATION_1):
    os.mkdir(LOCATION_1)

DATASET_FILES_2 = glob(os.path.join(LOCATION_1, "*.npy"))[:100000]

LOCATION_2 = os.path.abspath("D:/data/folkmagenta_3")
if not os.path.exists(LOCATION_2):
    os.mkdir(LOCATION_2)

VOCAB_PATH = os.path.join(LOCATION_2, 'vocab.npy')


if __name__ == "__main__":
    # first round
    # encode in 1d vector of magenta format
    print('first round: encoding into 1d magent format...')
    for f in tqdm(DATASET_FILES):
        seq = encode_into_magenta(f, folk_encoder)
        # slide window of 8 bars across the sequence
        sequences_available = compute_sequences_available(
            len(seq), MAX_SEQ_LEN, EVENTS_PER_BAR)

        for seq_i in range(sequences_available):
            seq_i_start = seq_i * EVENTS_PER_BAR
            seq_i_end = (seq_i + NR_BARS) * EVENTS_PER_BAR
            sequence = seq[seq_i_start:seq_i_end]

            assert len(sequence) == MAX_SEQ_LEN

            # save sequence
            split_name = f.split(os.path.sep)[-1]
            out_name = os.path.join(
                LOCATION_1, '%s_%s.npy' % (split_name, seq_i))
            np.save(out_name, sequence)

    # extract vocab
    print('second step: extracting vocab...')
    data = np.zeros((len(DATASET_FILES_2), MAX_SEQ_LEN), dtype='uint8')

    for i, f in enumerate(tqdm(DATASET_FILES_2)):
        seq = np.load(f).astype('uint8')
        data[i] = seq

    vocab = np.unique(data)
    np.save(VOCAB_PATH, vocab)
    print('wrote vocab to ', VOCAB_PATH)

    print('third step: one-hot encode using vocab...')
    for seq_i in tqdm(range(len(data))):
        seq = data[seq_i]
        new_seq = one_hot_encode_from_magenta(seq, VOCAB_PATH)
        seq_path = os.path.join(LOCATION_1, 'seq_%s' % seq_i)
        np.save(seq_path, new_seq)
