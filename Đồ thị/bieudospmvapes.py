from matplotlib import pyplot as plt
import numpy as np

pesphobert = 0.469156960489928
spmphobert = 0.470260009903743

pessimcse = 0.743325782926276
spmsimcse = 0.760326143741625

pesbkai = 0.663026766551342
spmbkai = 0.673643737041581

pesw2v = 0.500882109862454
spmw2v = 0.496649891933491

loaiss = np.array(["PhoBERT", "SimCSE", "BKai", "W2V"])
Pearson = np.array([pesphobert, pessimcse, pesbkai, pesw2v])
Spearman = np.array([spmphobert, spmsimcse, spmbkai, spmw2v])

bar_width = 0.35
index = np.arange(len(loaiss))

plt.bar(index, Pearson, bar_width, color='#a1def5', label='Pearson')
plt.bar(index + bar_width, Spearman, bar_width, color='#FFFF66', label='Spearman')

plt.xlabel("Phương pháp")
plt.ylabel("Độ tương quan")
plt.title("Thu thập bản đánh giá 1000 văn bản")
plt.xticks(index + bar_width / 2, loaiss)
plt.legend()

plt.show()