# import torch
#
# print("CUDA Available:", torch.cuda.is_available())
# print("Num GPUs Available:", torch.cuda.device_count())
import matplotlib.pyplot as plt

# 示例数据
data = [ [0.631578947368421, 0.631578947368421, 0.5, 0.5789473684210527, 0.6578947368421053], [0.3157894736842105, 0.368421052631578, 0.657894736842105, 0.631578947368421, 0.7631578947368421],
         [0.684210526315789, 0.6842105263157895, 0.605263157894736, 0.60526315789473, 0.52631578947368] ,[0.605263157894736,  0.5789473684210527, 0.684210526315789, 0.5263157894736842,0.3684210526315789],[0.76315789473,0.76315789473,0.7368421052631579,0.631578947368421,0.7925]]

# 创建箱线图
plt.boxplot(data)

# 设置标题和轴标签
plt.title('Accuracy Box Plot')
plt.ylabel('Value')
labels = ['gray', 'gray_deno', 'color_deno','enh_deno','all']
plt.xticks([1, 2, 3,4,5], labels)

# 显示图形
plt.show()

# import matplotlib.pyplot as plt
#
# # 示例数据
# data = [0.710526, 0.6052631578, 0.6578947, 0.631578947, 0.57894736, 0.6052631 ]
#
# # 绘制箱线图
# plt.boxplot(data)
# plt.title('Boxplot')
# plt.ylabel('Value')
# plt.show()
