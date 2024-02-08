import argparse
import json

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Dataset preprocessing program.')
parser.add_argument('--record', type=str, required=True, help='Record file for storing results.')
args = parser.parse_args()

# 获取分类器的预测结果和真实标签
with open(args.record, 'r') as f:
    result = json.load(f)
y_true = result['label']
y_score = result['pred']

# 计算ROC曲线和AUC
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()