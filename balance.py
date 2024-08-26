import matplotlib.pyplot as plt

# 假设我们有一个数据集，其中包含四个类别，每个类别的样本数量如下：
sample_counts = {'Control': 420, 'Concordant': 360, 'Discordant': 360}

# 提取类别名称和对应的样本数量
classes = list(sample_counts.keys())
counts = list(sample_counts.values())

# 创建柱状图
plt.figure(figsize=(8, 6))
plt.bar(classes, counts, color=['blue', 'green', 'purple'])

# 添加标题和标签
plt.title('The Distribution of Different Categories in Training Dateset')
plt.xlabel('Categories')
plt.ylabel('Number of Samples')

# 显示图形
plt.show()
