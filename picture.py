
#折线图
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# 模型指标数据（请替换为你的真实结果）
# ===============================
metrics = ['Accuracy', 'Precision', 'F1', 'AUC','AUPR','MCC','Spe']

# model_results = {
#     'GIP_CC+GIP_DD': [0.9273, 0.9027,  0.9301, 0.9813,0.9786,0.8581,0.8937],
#     'GIP_CC+SIM_DD ': [0.9366, 0.9169, 0.9385,  0.9837,0.9787,0.8757,0.9108],
#     'SIM_CC+GIP_DD':[0.5931,0.6115,0.5841,0.6363,0.6385,0.1903,0.6090],
#     'GIP_CD+SIM_DD+GIP_DC': [0.9390, 0.9307,  0.9398, 0.9831,0.9842,0.8791,0.9281],
#     'GIP_CD+SIM_DD+GIP_DC+SIM_CC': [ 0.8763, 0.8487, 0.8817, 0.9518,0.9497,0.7581,0.8324]
# }

# model_results = {
#     'Del_SIM_DD': [0.9273,0.9301 ,  0.9027, 0.9813,0.9786,0.8581,0.8937],
#     'Del_GIP_DC': [0.9366, 0.9385,0.9169 ,  0.9837,0.9787,0.8757,0.9108],
#     'Del_CNN_Auto': [0.8615, 0.8307,0.8668, 0.9372 ,0.9226,0.7342,0.8092],
#     'Rep_CNN(DNN)': [0.9218, 0.8916, 0.9251, 0.9766,0.9715,0.8476,0.8812],
#     'Our': [0.9390, 0.9307,  0.9398, 0.9831,0.9842,0.8791,0.9281]
# }

model_results = {
    'CNNE+SVM': [0.8060,0.8212,	0.8011,	0.8951,	0.9065,	0.6130,	0.8295],
    'CNNE+RF': [0.7488,	0.7485,	0.7486,	0.8284,	0.8510,	0.4995,	0.7465],
    'CNNE+XGboost': [0.7575,0.7527,	0.7599,	0.8504,	0.8552,	0.5171,	0.7450],
    'CNNE+DNN': [0.9218,0.8916,0.9251,0.9766,0.9715,0.8476,	0.8812],
    'CNNE+GRU': [0.8983,0.8704,	0.9019,	0.9542,0.9428,0.7990,0.8607],
    'Our': [0.9390, 0.9307,  0.9398, 0.9831,0.9842,0.8791,0.9281]
}

# ===============================
# 绘图
# ===============================
plt.figure(figsize=(8, 5))
x = np.arange(len(metrics))

# 仅使用颜色区分模型
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#DC143C','#A020F0','#FFD700']  # 蓝、橙、绿、紫
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#A020F0','#DC143C','#FFD700','#000080','#9467bd']
for i, (model_name, values) in enumerate(model_results.items()):
    plt.plot(
        x, values,
        marker='o',
        color=colors[i % len(colors)],
        linewidth=2,
        label=model_name
    )

# ===============================
# 图形美化
# ===============================
plt.xticks(x, metrics, fontsize=11)
plt.yticks(fontsize=11)
plt.ylim(0.1, 1.0)
plt.xlabel('Evaluation Metrics', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Classifier Comparison', fontsize=13, pad=12, fontweight='bold')

plt.legend(title='Model', fontsize=10, loc='lower right')
plt.grid(alpha=0.3)

plt.tight_layout()

# ===============================
# 保存图像
# ===============================
plt.savefig('Classifier Comparison_Performance_Comparison.png', dpi=600, bbox_inches='tight')
plt.savefig('Classifier Comparison_Performance_Comparison.pdf', bbox_inches='tight')

plt.show()



# #柱状图
# import matplotlib.pyplot as plt
# import numpy as np
#
# # ===============================
# # 模型指标数据（请替换为你的真实结果）
# # ===============================
# metrics = ['Accuracy', 'Precision', 'F1', 'AUC','AUPR','MCC','Spe']
#
# # 模型1与模型2的结果
# model1_name = 'DCDA'
# model2_name = 'Our'
#
# model1_scores = [0.8395, 0.7915, 0.8523, 0.9504,0.9572,0.6917,0.7539]
# model2_scores = [0.8918,0.8578,0.8971,0.9674,0.9727,0.7874,0.8430]
#
# # ===============================
# # 绘制柱状图
# # ===============================
# x = np.arange(len(metrics))  # 横轴位置
# width = 0.35  # 柱子宽度
#
# plt.figure(figsize=(8, 5), facecolor='white')  # 背景设为白色
#
# # 两个模型的柱子
# bar1 = plt.bar(x - width/2, model1_scores, width, label=model1_name, color='#1f77b4')
# bar2 = plt.bar(x + width/2, model2_scores, width, label=model2_name, color='#ff7f0e')
#
# # ===============================
# # 在柱子上显示具体数值
# # ===============================
# for bars in [bar1, bar2]:
#     for bar in bars:
#         yval = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.2f}',
#                  ha='center', va='bottom', fontsize=9)
#
# # ===============================
# # 图形美化
# # ===============================
# plt.xticks(x, metrics, fontsize=11)
# plt.yticks(fontsize=11)
# plt.ylim(0.7, 1.0)
# plt.xlabel('Evaluation Metrics', fontsize=12)
# plt.ylabel('Score', fontsize=12)
#
# # 图名（论文图标题）
# plt.title('Existing Optimal Research Comparison',
#           fontsize=13, pad=12, fontweight='bold')
#
# # 图例：放左上角 + 背景透明
# plt.legend(
#     title='Circ2Disease',
#     fontsize=10,
#     loc='upper left',
#     frameon=False  # 背景透明
# )
#
# plt.grid(axis='y', alpha=0.3)
# plt.tight_layout()
#
# # ===============================
# # 保存图像（背景为白色）
# # ===============================
# plt.savefig('Optimal_Research_Comparison_2.png', dpi=600, bbox_inches='tight', transparent=False)
# plt.savefig('Optimal_Research_Comparison_2.pdf', bbox_inches='tight', transparent=False)
#
# plt.show()
#
