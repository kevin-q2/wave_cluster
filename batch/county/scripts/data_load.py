import sys
import numpy as np
import pandas as pd
sys.path.append("wave_cluster")
import wave_cluster as wc


# This script's purpose is to load the Covid-19 dataset!
# Only input parameter is to decide whether or not to drop a larger subset of the locations

def load_data():
    # Import data
    index = pd.read_csv("data/index.csv", index_col = 0)
    demographics = pd.read_csv("data/demographics.csv", index_col = 0)
    
    data = pd.read_csv("data/us_county_daily.csv", index_col = 0)
    data = data.iloc[40:,:]
    population = demographics.loc[data.columns,"population"]
    
    data = data.fillna(0)
    norm_data = data.apply(lambda x: x/population[x.name])
    data = norm_data * 100000 # cases per 1000
    
    # Smoothing!
    front = 7
    back = 7
    smooth_data = wc.window_average(data.to_numpy(), front = front, back = back)
    smooth_data = wc.window_average(smooth_data, front = front, back = back)
    smooth_data = wc.window_average(smooth_data, front = front, back = back)
    smoothed_index = data.index[3*front:-3*back]
    data = pd.DataFrame(smooth_data, index = smoothed_index, columns = data.columns)
    data[data < 0] = 0
    
    return data
