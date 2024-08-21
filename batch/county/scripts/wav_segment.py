import sys
import os
import numpy as np
import pandas as pd
import time
from multiprocessing import Pool
import multiprocessing as mp

sys.path.append("batch/country/scripts/")
from data_load import load_data
sys.path.append("wave_cluster")
import wave_cluster as wc
sys.path.append("")
import wavefinder

data = load_data()
locations = data.columns

wavefinder_cuts = np.zeros((10, len(data.columns)))
wavefinder_cuts[:] = np.nan

for c in range(len(data.columns)):
    cname = data.columns[c]
    vec = data.loc[:,cname]
    vec.index = range(len(vec))
    wavelister = wavefinder.wavelist.WaveList(vec, 'CT', t_sep_a = 50, prominence_threshold = 10, prominence_height_threshold = 0)
    seg = wavelister.peaks_sub_c
    seg = [0] + list(seg.loc[seg.peak_ind == 0].location) + [data.shape[0]]
    wavefinder_cuts[:len(seg), c] = seg
    
wavefinder_cuts = pd.DataFrame(wavefinder_cuts, columns = data.columns)
wavefinder_cuts.to_csv("batch/county/data/wav_cuts.csv")