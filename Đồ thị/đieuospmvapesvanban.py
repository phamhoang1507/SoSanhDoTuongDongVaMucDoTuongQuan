from matplotlib import pyplot as plt
import numpy as np

pesphobert = 0.7694349838648114
spmphobert = 0.7866976408972604

pessimcse =  0.7604915335774668
spmsimcse = 0.755127142601396

# pesbkai = 0.663026766551342
# spmbkai = 0.673643737041581
#
# pesw2v = 0.500882109862454
# spmw2v = 0.496649891933491

# loaiss = np.array(["PhoBERT", "SimCSE", "BKai", "W2V"])
# Pearson = np.array([pesphobert, pessimcse, pesbkai, pesw2v])
# Spearman = np.array([spmphobert, spmsimcse, spmbkai, spmw2v])

loaiss = np.array(["PhoBERT-V2", "SimCSE"])
Pearson = np.array([pesphobert, pessimcse])
Spearman = np.array([spmphobert, spmsimcse])

bar_width = 0.35 # Giảm bar_width để cách nhau hơn
index = np.arange(len(loaiss))

plt.bar(index, Pearson, bar_width, color='#a1def5', label='Pearson')
plt.bar(index + bar_width, Spearman, bar_width, color='#FFFF66', label='Spearman')

plt.xlabel("Phương pháp")
plt.ylabel("Độ tương quan")
plt.title("Biểu đồ độ tương quan Pearson và Spearman của PhoBert và SimCSE cho 60 cặp văn bản")
plt.xticks(index + bar_width / 2, loaiss)
plt.legend()

plt.show()
