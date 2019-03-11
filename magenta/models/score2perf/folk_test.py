import glob
import sys
import numpy as np
sys.path.append("D:\\data\\magenta-1.0.2\\")
import pypianoroll

from music_encoders import MidiPerformanceEncoder

encoder = MidiPerformanceEncoder(100, 1, 40, 99)

fnames = glob.glob("D:\\data\\folkdataset\\*.mid")[:2]
for fname in fnames:
    seq = encoder.encode(fname)

    pianoroll_original = pypianoroll.Multitrack(fname).tracks[0].pianoroll
    mid = encoder.decode(seq)
    # print(mid)
    pianoroll_encoded = pypianoroll.Multitrack(mid).tracks[0].pianoroll

    equality = pianoroll_encoded == pianoroll_original
    print(fname)
    print(mid)


# test padding effect
for i, fname in enumerate(fnames):
    pianoroll_original = pypianoroll.Multitrack(fname).tracks[0].pianoroll

    seq = encoder.encode(fname)
    for _ in range(50):
        seq.append(0)

    mid = encoder.decode(seq, strip_extraneous=True)
    pianoroll_encoded = pypianoroll.Multitrack(mid).tracks[0].pianoroll

    equality = pianoroll_encoded == pianoroll_original
    print(fname)
    print(mid)



print('ALL GOOD')
