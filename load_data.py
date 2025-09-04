import pandas as pd
import matplotlib.pyplot as plt

# 读取 Excel 文件
df = pd.read_excel("D:\Mathmatical_Modeling\CUMCM2025\CUMCM2025Problems\C题\附件.xlsx")

# 查看前几行数据
print(df.head())

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

axes.scatter(x = df["孕妇BMI"], y = df["Y染色体浓度"], s = 20)
plt.show()
