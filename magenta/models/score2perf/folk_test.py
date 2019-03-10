import glob
import sys
import numpy as np
sys.path.append("D:\\data\\magenta-1.0.2\\")
import pypianoroll

from music_encoders import MidiPerformanceEncoder

encoder = MidiPerformanceEncoder(100, 32, 40, 99)

fnames = glob.glob("D:\data\folkdataset\*.mid")[:10]
for fname in fnames:
    seq = encoder.encode(fname)

    pianoroll_original = pypianoroll.Multitrack(fname).tracks[0].pianoroll

    # print(seq)

    mid = encoder.decode(seq)
    # print(mid)
    pianoroll_encoded = pypianoroll.Multitrack(fname).tracks[0].pianoroll

    equality = pianoroll_encoded == pianoroll_original
    assert False not in equality

print('ALL GOOD')
