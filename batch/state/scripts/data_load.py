import sys
import numpy as np
import pandas as pd
sys.path.append("../wave_cluster")
import wave_cluster as wc


# This script's purpose is to load the Covid-19 dataset!
# Only input parameter is to decide whether or not to drop a larger subset of the locations

def load_data():
    # Import indexing and related demographic data
    index = pd.read_csv("data/index.csv", index_col = 0)
    demographics = pd.read_csv("data/demographics.csv", index_col = 0)

    # US state-level daily case data (# new cases every day)
    # Dropping the first 39 timestamps during which reporting practices were still being developed
    # and dropping a few of the territories with irregular data
    data = pd.read_csv("data/us_state_daily.csv", index_col = 0)
    data = data.iloc[39:,:]
    dr = ['US_VI', 'US_MP', 'US_GU', 'US_AS', 'US_DC']
    data = data.drop(dr, axis = 1)


    # Normalization of each location's time-series to be cases per 100,000 persons (typical in the literature)
    population = demographics.loc[data.columns,"population"]
    norm_data = data.apply(lambda x: x/population[x.name])
    data = norm_data * 100000 # cases per 100,000


    # Windowed average smoothing of each time-series 
    # Using 7 day average with 3 days in front and behind burrent time-stamp
    front = 7
    back = 7
    
    # Double averaging procedure (gets rid of sharp fluctuations)
    smooth_data = wc.window_average(data.to_numpy(), front = front, back = back)
    smooth_data = wc.window_average(smooth_data, front = front, back = back)
    smooth_data = wc.window_average(smooth_data, front = front, back = back)
    smoothed_index = data.index[front*3:-back*3]
    data = pd.DataFrame(smooth_data, index = smoothed_index, columns = data.columns)

    # remove any negative entries (reporting errors still present after smoothing)
    data[data < 0] = 0
    
    return data