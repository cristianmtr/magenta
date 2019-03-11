from glob import glob
import os
import sys

from my_encoder import encoder  # make sure we use the same settings
import numpy as np
from tqdm import tqdm

sys.path.append("D:\\data\\magenta-1.0.2\\")
sys.path.append("D:\\data\\thesis_model2\\")

from utils import compute_sequences_available

EVENTS_PER_BAR = 24
NR_BARS = 8
MAX_SEQ_LEN = EVENTS_PER_BAR * NR_BARS


DATASET_LOC = os.path.abspath("D:\\data\\folkdataset\\")
DATASET_GLOB = os.path.join(DATASET_LOC, "*.mid")
DATASET_FILES = glob(DATASET_GLOB)
NEW_LOCATION = os.path.abspath("D:\\data\\folkmagenta_2\\")
if not os.path.exists(NEW_LOCATION):
    os.mkdir(NEW_LOCATION)


for f in tqdm(DATASET_FILES):
    seq = encoder.encode(f)
    # slide window of 8 bars across the sequence
    sequences_available = compute_sequences_available(len(seq), MAX_SEQ_LEN, EVENTS_PER_BAR)

    for seq_i in range(sequences_available):
        seq_i_start = seq_i * EVENTS_PER_BAR
        seq_i_end = (seq_i + NR_BARS) * EVENTS_PER_BAR
        sequence = seq[seq_i_start:seq_i_end]

        assert len(sequence) == MAX_SEQ_LEN

        # save sequence
        split_name = f.split(os.path.sep)[-1]
        out_name = os.path.join(NEW_LOCATION, '%s_%s.npy' % (split_name, seq_i))
        np.save(out_name, sequence)    
