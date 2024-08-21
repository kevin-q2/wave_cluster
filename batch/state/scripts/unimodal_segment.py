import sys
import os
import numpy as np
import pandas as pd
import time
from multiprocessing import Pool
import multiprocessing as mp

sys.path.append("batch/state/scripts/")
from data_load import load_data
sys.path.append("../wave_cluster")
import wave_cluster as wc

job_id = int(os.getenv('SGE_TASK_ID'))
#cpu_count = int(os.getenv('NSLOTS'))
cpu_count = 16

data = load_data()
locations = data.columns

total_jobs = 1
per = data.shape[1]

if job_id == total_jobs:
    locations = locations = data.columns[per*(job_id - 1):]
else:
    locations = data.columns[per*(job_id - 1):per*job_id]

data = data.loc[:, locations]

# parameters for manual/custom results
max_segments = 10
threshold = 0.005
segment_size = 30 # keep in mind that this is really half of what the actual wave segments will look like!!
                # because unimodal computes both an increasing and decreasing part of the wave. 

est_waves = np.zeros(data.shape)
est_cuts = np.zeros((2*max_segments, data.shape[1]))
elbows = np.zeros((len(range(2,max_segments)), data.shape[1]))


for l in range(len(locations)):
    loc = locations[l]
    vec = data.loc[:,loc].to_numpy()
    increasing_kmodal_error, decreasing_kmodal_error = wc.compute_kmodal_error_table(vec, segment_size = 1, cpu_count = 16)
    dp = wc.dynamic_segment(increasing_kmodal_error, segments = 2*max_segments, segment_size = segment_size, error_table2 = decreasing_kmodal_error)
    dp.fill_table()
    
    def unimodal_error(cuts):
        errors,uni_wave = wc.unimodal_fit(vec).generate(cuts)
        rel_error = np.linalg.norm(vec - uni_wave)/np.linalg.norm(vec)
        return rel_error
    
    rels = dp.tester(np.array(range(2,max_segments))*2, unimodal_error)
    elbows[:,l] = rels
    
    chosen_idx = wc.elbow(rels, threshold)
    chosen_segs = list(range(2,max_segments))[chosen_idx]    
    #print(chosen_segs)
    dp.backtrack(2*chosen_segs)
    errors,uni_wave = wc.unimodal_fit(vec).generate(dp.cuts)
    
    est_waves[:,l] = uni_wave
    est_cuts[:len(dp.cuts),l] = dp.cuts
    
    
est_waves = pd.DataFrame(est_waves, columns = locations, index = data.index)
est_cuts = pd.DataFrame(est_cuts, columns = locations)

est_cuts.replace(0, np.nan, inplace=True)
est_cuts.iloc[0,:] = np.zeros(est_cuts.shape[1])


if total_jobs > 1:
    est_cuts.to_csv("batch/state/data/unimodal_cuts" + str(job_id) + ".csv")
else:
    est_cuts.to_csv("batch/state/data/unimodal_cuts.csv")
