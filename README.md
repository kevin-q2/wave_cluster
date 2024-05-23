# Covid Wave Clustering

In this repository is an implementation of a infection wave clustering methodology which has code to 

1. Split infection time-series into wave-like segments
2. Cluster those waves in a way that respects their timing within a global time period

We consider a wave to be a short period of pronounced infection activity, and argue that modeling and clustering waves 
gives a good way to understand the evolving spatio-temporal patterns of a disease or virus.

## Installation 
Here's how to get started:
```
# *Recommended* Clone the repository
git clone https://github.com/your-username/your-repository.git

# OR install as a library
pip install git+https://github.com/kevin-q2/wave_cluster.git#egg=wave_cluster
```

We recommend simply cloning the entire repository in order to get all of the data and example notebooks, but installing just the 
code as a library is possible as well. 

## Data
Integral to our analyses and experiments is the data collected and provided in `data/` with attribution given to the following sources:
  1. [Google's Covid-19 Covid Repository](https://github.com/GoogleCloudPlatform/covid-19-open-data) records daily infection records from locations
     around the world. To them we attribute daily infection records for US states in `us_state_daily.csv` and from countries around the world in `country_daily.csv`.
     The data from `index.csv`, `demographics.csv`, and `geography.csv` also contain useful organizational and auxiliary information associated with this data.
  2. [Oxford's Covid Government Response Tracker](https://github.com/OxCGRT/covid-policy-dataset) records daily information about each locations
     stringency or proactive responses to containment of the virus. We provide a this data for US states and a set of European countries in
     `state_containment_health.csv` and `country_containment_health.csv` respectively.
  3. From the US census we also collect information on each state's [centers of population](https://www.census.gov/geographies/reference-files/time-series/geo/centers-population.html) in
     `state_centers.csv`, their [density](https://www.census.gov/data/tables/time-series/dec/density-data-text.html) of population in `us_state_density.csv` and their geographical
     [boundary lines](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) for visualization found in shape
     files in `visualization/tl_rd22_us_state/`
  4. Finally we also include some shape files for [country visualization](https://public.opendatasoft.com/explore/dataset/world-administrative-boundaries/information/?flg=en-us&location=10,22.37175,114.10565&basemap=jawg.light) fand  
     
