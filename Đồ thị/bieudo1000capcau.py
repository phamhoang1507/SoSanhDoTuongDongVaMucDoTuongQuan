

import matplotlib.pyplot as plt
import numpy as np

diem = np.array([0, 1, 2, 3, 4])
soluong = np.array([216, 201, 146, 202, 234])

# # Mức mới
new_levels = np.array([0, 50, 100, 150, 200, 250,300])

# Tính toán số lượng mới dựa trên mức mới
new_soluong = soluong / 50 * new_levels[1]

plt.bar(diem, new_soluong, color=['red', 'purple'])
plt.xlabel("Đánh Giá")
plt.ylabel("Số Lượng")
plt.title("Thu Thập Bản Đánh Giá 1000 Văn Bản")

# Chỉ định các dải giá trị trên trục y
plt.yticks(new_levels)

plt.show()