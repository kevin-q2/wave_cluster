import sys
import os
import numpy as np
import pandas as pd
import time
from multiprocessing import Pool
import multiprocessing as mp

sys.path.append("batch/country/scripts/")
from data_load import load_data
sys.path.append("../wave_cluster")
import wave_cluster as wc


# I use this as a way of splitting the task up into different jobs 
# (i.e. I usually run these remotely on a compute cluster and split up the tasks since they can be intensive)
job_id = int(os.getenv('SGE_TASK_ID'))
#cpu_count = int(os.getenv('NSLOTS'))
cpu_count = 16


data = load_data()
EU = ['AD', 'AL', 'AM','AT','AZ','BA', 'BE', 'BG','BY', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR',
     'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 
     'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE', 'GB', 'GE', 'IS', 'KZ', 'LI', 'MD', 'MC', 'ME',
     'MK', 'NO', 'RU', 'SM', 'RS', 'CH', 'TR', 'UA']

locations = np.sort(EU)
data = data.loc[:, locations]
# These european locations don't have associated containment health data
data = data.drop(['AM','MK', 'ME'], axis = 1)
locations = data.columns

# I'm reading in the unimodal cuts because we actually use these for help with the SIR ones. 
# Specifically for each location we just use the same number of segments that unimodal does instead of trying to optimize
unimodal_cuts = pd.read_csv("batch/country/data/unimodal_cuts2.csv", index_col = 0)

# total number of jobs to use
# number of locations to compute per one job
total_jobs = 10
per = data.shape[1]//total_jobs
if job_id == total_jobs:
    locations = data.columns[per*(job_id - 1):]
else:
    locations = data.columns[per*(job_id - 1):per*job_id]
    
data = data.loc[:, locations]


# parameters for manual/custom results
max_segments = 10 # maximum number of segments to split the time-series into 
segment_size = 60 # minimum possible segment size 
threshold = 0.005 # threshold value used for selecting the number of segments used 

# storing computed waves, cuts, elbow curves for fitting
est_waves = np.zeros((data.shape[0], len(locations)))
est_cuts = np.zeros((max_segments + 1, len(locations)))
elbows = np.zeros((len(range(2,max_segments)), len(locations)))

print('segmenting!')
for l in range(len(locations)):
    loc = locations[l]
    vec = data.loc[:,loc].to_numpy()
    vec_pop = 100000 # use population of 100,000 since we already normalized ot this!
    error_table = wc._sir_fit.compute_sir_error_table(vec, vec_pop, initialization='auto', segment_size = segment_size, cpu_count = cpu_count)
    
    S = wc.dynamic_segment(error_table, max_segments, segment_size)
    S.fill_table()
    
    # choosing # of segments!
    def sir_error(cuts):
        errors,wave,parameters = wc.sir_fit(vec, vec_pop, initialization = 'auto').generate(cuts)
        rel_error = np.linalg.norm(vec - wave)/np.linalg.norm(vec)
        return rel_error

    rels = S.tester(np.array(range(2,max_segments)), sir_error)
    elbow_res = wc.elbow(rels, threshold)
    if elbow_res is None:
        print('Elbow fail ')
        print(loc)
        print()
        elbow_res = 3
        
    elbows[:,l] = rels
    
    # choosing # of segments automatically!
    #chosen_segs = list(range(2,max_segments))[int(elbow_res)]    

    # Here we decided to use the same number of segments that was chosen by unimodal segmentation
    ucuts = unimodal_cuts.loc[:,loc].to_numpy()
    chosen_segs = int((len(ucuts[~np.isnan(ucuts)]) - 1) / 2)
    
    S.backtrack(chosen_segs)
    errors, wave_est, parameters = wc.sir_fit(vec, vec_pop, initialization = 'auto').generate(S.cuts)
    
    est_waves[:,l] = wave_est
    est_cuts[:len(S.cuts),l] = S.cuts
    
    
# store and save
est_waves = pd.DataFrame(est_waves, columns = locations, index = data.index)
est_cuts = pd.DataFrame(est_cuts, columns = locations)

est_cuts.replace(0, np.nan, inplace=True)
est_cuts.iloc[0,:] = np.zeros(est_cuts.shape[1])

if total_jobs > 1:
    est_waves.to_csv("batch/country/data/sir_waves_" + str(job_id) + ".csv")
    est_cuts.to_csv("batch/country/data/sir_cuts_" + str(job_id) + ".csv")
else:
    est_waves.to_csv("batch/country/data/sir_waves.csv")
    est_cuts.to_csv("batch/country/data/sir_cuts.csv")


#elbows = pd.DataFrame(elbows, columns = locations)
#elbows.to_csv('batch/country/data/sir_elbows.csv')


