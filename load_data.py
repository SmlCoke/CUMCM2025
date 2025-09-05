import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "main":
    # 读取 Excel 文件
    df = pd.read_excel("D:\Mathmatical_Modeling\CUMCM2025\CUMCM2025Problems\C题\附件.xlsx")

    # 查看前几行数据
    print(df.head())

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

    axes.scatter(x = df["孕妇BMI"], y = df["Y染色体浓度"], s = 20)
    plt.show()


    # 绘制二维直方图
    # 绘制二维直方图（热力图）
    x = df["孕妇BMI"]
    y = df["Y染色体浓度"]
    plt.hist2d(x, y, bins=30, cmap='Blues')  # bins可调，cmap可选
    plt.colorbar(label='计数')
    plt.xlabel('BMI')
    plt.ylabel('Y')
    plt.title('Histogram')
    plt.show()