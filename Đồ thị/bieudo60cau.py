

import matplotlib.pyplot as plt
import numpy as np

diem = np.array([0, 1, 2, 3, 4,5,6])
soluong = np.array([18, 2, 5,11, 5, 12,7])

# # Mức mới
new_levels = np.array([0, 5, 10, 15, 20, 25])

# Tính toán số lượng mới dựa trên mức mới
new_soluong = soluong / 5 * new_levels[1]

plt.bar(diem, new_soluong, color=['red', 'purple'])
plt.xlabel("Đánh Giá")
plt.ylabel("Số Lượng")
plt.title("Thu Thập Bản Đánh Giá 60 Cặp Văn Bản")

# Chỉ định các dải giá trị trên trục y
plt.yticks(new_levels)

plt.show()