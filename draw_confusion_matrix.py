
import numpy as np
import matplotlib.pyplot as plt
import itertools
# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    # print(cm)

    # plt.figure(figsize=(6, 6), dpi=100)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontdict={'family' : 'Times New Roman', 'size'   : 10})
    plt.yticks(tick_marks, classes, fontdict={'family' : 'Times New Roman', 'size'   : 10})
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 fontdict={'family' : 'Times New Roman', 'size'   : 10},
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    cb= plt.colorbar()
    for l in cb.ax.yaxis.get_ticklabels():
        l.set_family('Times New Roman')
    plt.ylabel('Ground Truth', fontdict={'family' : 'Times New Roman', 'size'   : 12})
    plt.xlabel('Predicted Label', fontdict={'family' : 'Times New Roman', 'size'   : 12})
    plt.tight_layout()
    # plt.show()
    plt.savefig("c_m.pdf", dpi=300, format="pdf")

if __name__=='__main__':
    cnf_matrix = np.load('all_results/RandLA-Net2_Toronto-3D_xyz/predict_results/train_best/confusion_matrix.npy')
    attack_types = ['Road', 'Rd mrk.', 'Natural' ,'Building', 'Util. line', 'Pole', 'Car', 'Fence']
    plot_confusion_matrix(cnf_matrix, classes=attack_types, normalize=True, title='Normalized confusion matrix')