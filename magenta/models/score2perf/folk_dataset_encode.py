import numpy as np
from tqdm import tqdm
import os
import sys
from glob import glob
sys.path.append("D:\\data\\magenta-1.0.2\\")

from music_encoders import MidiPerformanceEncoder
encoder = MidiPerformanceEncoder(100, 32, 44, 97) # precomputed min and max, 32 and 100 taken from paper

DATASET_LOC = os.path.abspath("D:\\data\\folkdataset\\")
DATASET_GLOB = os.path.join(DATASET_LOC, "*.mid")
DATASET_FILES = glob(DATASET_GLOB)
NEW_LOCATION = os.path.abspath("D:\\data\\folkmagenta\\")


for f in tqdm(DATASET_FILES):
    seq = encoder.encode(f)
    seq_file = f.split(os.path.sep)[-1]
    seq_file = os.path.join(NEW_LOCATION, seq_file)
        
