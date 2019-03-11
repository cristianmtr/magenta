import tqdm
import numpy as np
from glob import glob
import os
from matplotlib import pyplot as plt
from my_encoder import encoder
import pypianoroll

DATASET_LOC = os.path.abspath("D:\\data\\folkdataset\\")
DATASET_GLOB = os.path.join(DATASET_LOC, "*.mid")
DATASET_FILES = glob(DATASET_GLOB)
NEW_LOCATION = os.path.abspath("D:\\data\\folkmagenta\\")

files = 200
DATASET_FILES = DATASET_FILES[:files]

lens = []

for fname in tqdm.tqdm(DATASET_FILES):
    one_bar = pypianoroll.Multitrack(fname).tracks[0].pianoroll[:96]
    midi_one_bar = pypianoroll.Multitrack(tracks=[pypianoroll.Track(one_bar)])
    midi_one_bar.write('test.mid')
    seq = encoder.encode('test.mid')
    lens.append(len(seq))


print(np.mean(lens), np.std(lens))
print(np.quantile(lens, 0.75)) # 24
plt.hist(lens)
plt.show()
