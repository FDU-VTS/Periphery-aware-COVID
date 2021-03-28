import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

# # 加载数据
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# # 将标签二值化
# y = label_binarize(y, classes=[0, 1, 2])
# # 设置种类
# n_classes = y.shape[1]

# # 训练模型并预测
# random_state = np.random.RandomState(0)
# n_samples, n_features = X.shape

# # shuffle and split training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,random_state=0)

# # Learn to predict each class against the other
# classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
#                                  random_state=random_state))
# y_score = classifier.fit(X_train, y_train).decision_function(X_test)
# print(y_score)
# print(np.array(y_test))
# exit()

# 计算每一类的ROC
def draw_roc(y_test,y_score,png_name,n_classes=2):
    y_test = label_binarize(y_test, classes=[0, 1])
    # y_test = label_binarize(y_test, classes=range(n_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw=2
    plt.switch_backend('agg')
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.4f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)
    print("micro_auc: ", roc_auc["micro"]*100.)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.4f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)
    print("macro_auc: ", roc_auc["macro"]*100.)

    colors = cycle(['red', 'darkorange', 'gold', 'yellowgreen'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.4f})'
                ''.format(i, roc_auc[i]))
        print(i, roc_auc[i]*100.)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('ROC curve of TFCNN-II(pre).')
    plt.legend(loc="lower right")
    plt.savefig('./'+ png_name +'.png')
    print('save auc!')
