import pandas as pd
import matplotlib.pyplot as plt
data1 = pd.read_csv("/Users/pabloalonso/OneDrive - University of Warwick/Dissertation/PDMP/Experiments/BLR_HorseshoePrior/d50/results/d50_slar/sparsity_hmc_d50_slar.csv")
data2 = pd.read_csv("/Users/pabloalonso/OneDrive - University of Warwick/Dissertation/PDMP/Experiments/BLR_HorseshoePrior/d50/results/d50_slar/sparsity_bk_d50_slar.csv")
data3 = pd.read_csv("/Users/pabloalonso/OneDrive - University of Warwick/Dissertation/PDMP/Experiments/BLR_HorseshoePrior/d50/results/d50_slar/sparsity_bps_d50_slar.csv")

data4 = pd.read_csv("/Users/pabloalonso/OneDrive - University of Warwick/Dissertation/PDMP/Experiments/BLR_HorseshoePrior/d50/results/d50_slar/perCorrect_hmc_d50_slar.csv")
data5 = pd.read_csv("/Users/pabloalonso/OneDrive - University of Warwick/Dissertation/PDMP/Experiments/BLR_HorseshoePrior/d50/results/d50_slar/sparsity_bk_d50_slar.csv")
data6 = pd.read_csv("/Users/pabloalonso/OneDrive - University of Warwick/Dissertation/PDMP/Experiments/BLR_HorseshoePrior/d50/results/d50_slar/sparsity_bps_d50_slar.csv")

plt.plot(data1['0'])
plt.plot(data2['0'])
plt.plot(data3['0'])


plt.plot(data4['0'])
plt.plot(data5['0'])
plt.plot(data6['0'])