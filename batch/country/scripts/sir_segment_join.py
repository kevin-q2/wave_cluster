import numpy 
import pandas as pd

sir_waves1 = pd.read_csv("../data/sir_waves1.csv", index_col = 0)
sir_cuts1 = pd.read_csv("../data/sir_cuts1.csv", index_col = 0)

sir_waves2 = pd.read_csv("../data/sir_waves2.csv", index_col = 0)
sir_cuts2 = pd.read_csv("../data/sir_cuts2.csv", index_col = 0)

sir_waves3 = pd.read_csv("../data/sir_waves3.csv", index_col = 0)
sir_cuts3 = pd.read_csv("../data/sir_cuts3.csv", index_col = 0)

sir_waves4 = pd.read_csv("../data/sir_waves4.csv", index_col = 0)
sir_cuts4 = pd.read_csv("../data/sir_cuts4.csv", index_col = 0)

sir_waves5 = pd.read_csv("../data/sir_waves5.csv", index_col = 0)
sir_cuts5 = pd.read_csv("../data/sir_cuts5.csv", index_col = 0)

sir_waves6 = pd.read_csv("../data/sir_waves6.csv", index_col = 0)
sir_cuts6 = pd.read_csv("../data/sir_cuts6.csv", index_col = 0)

sir_waves7 = pd.read_csv("../data/sir_waves7.csv", index_col = 0)
sir_cuts7 = pd.read_csv("../data/sir_cuts7.csv", index_col = 0)

sir_waves8 = pd.read_csv("../data/sir_waves8.csv", index_col = 0)
sir_cuts8 = pd.read_csv("../data/sir_cuts8.csv", index_col = 0)

sir_waves9 = pd.read_csv("../data/sir_waves9.csv", index_col = 0)
sir_cuts9 = pd.read_csv("../data/sir_cuts9.csv", index_col = 0)

sir_waves10 = pd.read_csv("../data/sir_waves10.csv", index_col = 0)
sir_cuts10 = pd.read_csv("../data/sir_cuts10.csv", index_col = 0)



sir_cuts = sir_cuts1.join([sir_cuts2, sir_cuts3, sir_cuts4, sir_cuts5, sir_cuts6, sir_cuts7, sir_cuts8, sir_cuts9, sir_cuts10])
sir_waves = sir_waves1.join([sir_waves2, sir_waves3, sir_waves4, sir_waves5, sir_waves6, sir_waves7, sir_waves8, sir_waves9, sir_waves10])

sir_waves.to_csv('../data/sir_waves.csv')
sir_cuts.to_csv('../data/sir_cuts.csv')
