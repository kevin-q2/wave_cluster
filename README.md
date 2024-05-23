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
Integral to our experiments is the data collected and provided in `data/`
