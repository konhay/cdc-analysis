# -*- coding:utf-8 -*-
import pandas as pd
import MySQLdb
import numpy as np
import matplotlib.pyplot as plt
import pylab as plot
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error
from sklearn.ensemble import AdaBoostClassifier
from sklearn import ensemble
from sklearn import metrics
from math import sqrt, fabs, exp
from .constant import *

#plt.style.use('fivethirtyeight')

COLUMNS = ['MALE',
         'FEMALE',
         'AGE',
         'EDUCA',
         'INCOME_ANN',
         'OCC_CODE',
         'ACTS',
         'OPEN_DAY',
         'CRED_LIMIT',
         'USE_RATE',
         'HI_PURCHSE',
         'OCT_COUNT',
         'DUE_CNT',
         'DUE_MA_CNT',
         'DET_CRE',
         'DUE_AMT',
         'PMT_AMT',
         'PMT_DAY',
         'CT_MP_Y',
         'CASH_CT',
         'CASH_AV',
         'M_PRECIOUS',
         'M_SPORTSCLUB',
         'M_FURNITURE',
         'M_AUTOSERVICE'];

def get_frame(sql):
    '''
    get credit card due data from mysql
    '''
    try:
        con = MySQLdb.connect('localhost', 'root', 'root', 'psbc', charset='utf8')
        cur = con.cursor()
        cur.execute(sql)
        data = cur.fetchall()
        frame = pd.DataFrame(list(data))
        frame.columns = COLUMNS
    except:
        frame = pd.DataFrame()
    # plt.pcolor(frame.iloc[:,:].corr())
    # plt.show()
    print("frame.shape:", frame.shape)
    return frame


def make_dataset(frame):
    '''
    make train and test dataset
    '''
    # 1.random over sample
    # from imblearn.over_sampling import RandomOverSampler
    # ros = RandomOverSampler(random_state=0)
    # X_resampled, y_resampled = ros.fit_sample(X, y)

    # 2.SMOTE(1:9)
    frame_0 = frame[frame[frame.columns[-1]] == 0]
    frame_1 = frame[frame[frame.columns[-1]] == 1]
    frame = frame_0.sample(int(1000*8.36)).append(frame_1.sample(1000))
    x = frame.iloc[:, 0:-1].values
    y = frame.iloc[:, -1].values
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.30, random_state=531)
    sm = SMOTEENN(ratio = 0.5)#(ratio = {1:2900, 0:5800})#or ratio = 0.5
    xTrain, yTrain = sm.fit_sample(xTrain, yTrain)

    return xTrain, yTrain, xTest, yTest


def rf_classifiter():
    '''
    RandomForestClassifier
    '''
    # RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
    #     max_depth=1, max_features='auto', max_leaf_nodes=None,
    #     min_impurity_decrease=0.0, min_impurity_split=None,
    #     min_samples_leaf=1, min_samples_split=2,
    #     min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
    #     oob_score=False, random_state=0, verbose=0, warm_start=False)

    frame = get_frame(SQL1)
    xTrain, yTrain, xTest, yTest = make_dataset(frame)

    forest = ensemble.RandomForestClassifier(
        n_estimators=100, criterion="gini", max_depth=1, random_state=0)
    forest.fit(xTrain,yTrain)
    score = forest.score(xTest,yTest)
    print("准确率:%.2f%%" %(score*100))

    forest_y_score = forest.predict_proba(xTest)
    forest_fpr1,forest_tpr1,_ = metrics.roc_curve(
        label_binarize(yTest,classes=(0,1,2)).T[0:-1].T.ravel(),forest_y_score.ravel())
    auc = metrics.auc(forest_fpr1, forest_tpr1)
    print("目标属性AUC值:%.2f%%" %auc)

    plt.figure(figsize=(8, 6), facecolor='w')
    plt.plot(forest_fpr1, forest_tpr1, c='r', lw=2, label=u'AUC=%.3f' % auc)
    plt.plot((0,1), (0,1), c='#a0a0a0', lw=2, ls='--')
    plt.xlim(-0.001, 1.001)
    plt.ylim(-0.001, 1.001)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate(FPR)', fontsize=16)
    plt.ylabel('True Positive Rate(TPR)', fontsize=16)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
    plt.title(u'RF ROC Curve', fontsize=18)
    plt.savefig("RF ROC Curve.png")
    plt.show()


def rf_regressor():
    '''
    RandomForestRegressor
    '''
    frame = get_frame(SQL1)
    xTrain, yTrain, xTest, yTest = make_dataset(frame)

    mseOos = []
    nTreeList = range(10, 250, 10) #10,200,10
    for iTrees in nTreeList:
        depth = None
        maxFeat = int(round(np.sqrt(xTrain.shape[1]-1)))
        wineRFModel = ensemble.RandomForestRegressor(n_estimators=iTrees, max_depth=depth, max_features=maxFeat, oob_score=False, random_state=531)
        wineRFModel.fit(xTrain,yTrain)
        prediction = wineRFModel.predict(xTest)
        mseOos.append(mean_squared_error(yTest, prediction))
    print("MSE:", mseOos[-1])

    ct, ct_1, ct_0 = [0, 0, 0]
    for i in range(len(prediction)):
        if abs(prediction[i] - yTest[i]) < 0.5: ct += 1
        if yTest[i] == 1 and prediction[i] >= 0.5: ct_1 += 1
        if yTest[i] == 0 and prediction[i] < 0.5: ct_0 += 1
    print("CORRECT RATE:", round(float(ct) / len(prediction), 3))
    print("TRUE CORRECT RATE:", round(float(ct_1) / len(filter(lambda x: x == 1, yTest)), 3))
    print("FALSE CORRECT RATE:", round(float(ct_0) / len(filter(lambda x: x == 0, yTest)), 3))

    plot.plot(nTreeList, mseOos)
    plot.xlabel('Number of Trees in Ensemble')
    plot.ylabel('Mean Squared Error')
    plot.show()

    wineNames = np.array(statement.names)
    featureImportance = wineRFModel.feature_importances_
    featureImportance = featureImportance / featureImportance.max()
    sorted_idx = np.argsort(featureImportance)
    barPos = np.arange(sorted_idx.shape[0]) + .5
    plot.barh(barPos, featureImportance[sorted_idx], align='center')
    plot.yticks(barPos, wineNames[sorted_idx])
    plot.xlabel('Variable Importance')
    plot.show()
    print("Variable Importance asc ordered:")
    print(wineNames[sorted_idx])


def make_dataset_pro(frame):
    '''
    make train and test dataset
    '''
    # no sampling
    # X = frame.iloc[:, 0:-1].values
    # y = frame.iloc[:, -1].values
    # xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.8, random_state=531)

    # smote
    # frame_0 = frame[frame[frame.columns[-1]] == 0]
    # frame_1 = frame[frame[frame.columns[-1]] == 1]
    # frame = frame_0.sample(int(7000*8.36)).append(frame_1.sample(7000))
    # X = frame.iloc[:, 0:-1].values
    # y = frame.iloc[:, -1].values
    # xSample, xTest, ySample, yTest = train_test_split(X, y, test_size=0.30, random_state=531)
    # sm = SMOTEENN(ratio = 0.25)
    # xTrain, yTrain = sm.fit_sample(xSample, ySample)

    # under sampling
    X = frame.iloc[:, 0:-1].values
    y = frame.iloc[:, -1].values
    xSample, xTest, ySample, yTest = train_test_split(X, y, test_size=0.8, random_state=531)
    rus = RandomUnderSampler(random_state=3, ratio=1)
    xTrain, yTrain = rus.fit_sample(xSample, ySample)

    return xTrain, yTrain, xTest, yTest


# 加载数据集
frame = get_frame(SQL1)
xTrain, yTrain, xTest, yTest = make_dataset(frame)

# 数据概览
pos, neg = 0,0
for i in yTrain:
    if i == 1:
        pos=pos+1
    else:
        neg=neg+1
print("total train:",len(yTrain), "neg:", neg, "pos:", pos, "ratio:", neg/pos)

# 初始化抽样权重
# w1 = 2
# sample_weight = np.ones(len(yTrain))
# for i in range(len(sample_weight)):
#     if yTrain[i] == 1:
#         sample_weight[i] =w1

# 作弊
# xTest = np.concatenate([xTest,xTrain])
# yTest = np.concatenate([yTest,yTrain])

#adaboost parameters
n_estimators_ab = 80             #default 50
learning_rate = 1.0              #default 1.0
algorithm = ["SAMME.R", "SAMME"] #default SAMME.R

#randomforest parameters
n_estimators_rf = 200            #default 10
max_depth = None                 #default None(means no limit)
max_features = 4                 #default None(sqrt)
criterion = ["gini", "entropy"]  #default SAMME.R

#training
#######################start########################
auc = []
accuracy = []
#nTreeList = range(200, 201, 1)
learnList = np.arange(0.6,0.7,0.1)
#featureList = range(4,5,1)
#depthList = range(10,101,10)

#for iTrees in nTreeList:
for iLearn in learnList:
#for iFeature in featureList:
#for iDepth in depthList:
    # rocksVMinesRFModel = ensemble.RandomForestClassifier(
    #     n_estimators=n_estimators_rf,
    #     criterion=criterion[0],
    #     max_depth=iDepth,
    #     max_features=max_features,
    #     oob_score=False,
    #     random_state=531)
    rocksVMinesRFModel = AdaBoostClassifier(
        n_estimators=n_estimators_ab,
        learning_rate=iLearn,
        algorithm=algorithm[0],
        random_state=531)
    rocksVMinesRFModel.fit(xTrain,yTrain)
    #rocksVMinesRFModel.fit(xTrain,yTrain,sample_weight)

    #Accumulate auc on test set
    prediction = rocksVMinesRFModel.predict_proba(xTest)
    aucCalc = roc_auc_score(yTest, prediction[:,1:2])
    auc.append(aucCalc)

    #accumulate accuracy by hanbing
    score = rocksVMinesRFModel.score(xTest, yTest)
    accuracy.append(score)

#######################end########################

#print last AUC
print('AUC', auc[-1])

#print last accuracy
print('accuracy', accuracy[-1])

#plot training and test errors vs number of trees in ensemble
# plot.plot(nTreeList, auc, color='blue')
# plot.xlabel('Number of Trees in Ensemble')
# plot.ylabel('Area Under ROC Curve - AUC')
# #plot.ylim([0.0, 1.1*max(mseOob)])
# plot.show()

# #plot accuracy
# plot.plot(nTreeList, accuracy, color='blue')
# plot.xlabel('Number of Trees in Ensemble')
# plot.ylabel('Accuracy')
# plot.show()

#plot training and test errors vs number of trees in ensemble
# plot.plot(learnList, auc, color='blue')
# plot.xlabel('Different Learning Rate')
# plot.ylabel('Area Under ROC Curve - AUC')
# #plot.ylim([0.0, 1.1*max(mseOob)])
# plot.show()
#
# #plot accuracy
# plot.plot(learnList, accuracy, color='blue')
# plot.xlabel('Different Learning Rate')
# plot.ylabel('Accuracy')
# plot.show()

#plot training and test errors vs number of trees in ensemble
# plot.plot(featureList, auc, color='blue')
# plot.xlabel('Number of Features')
# plot.ylabel('Area Under ROC Curve - AUC')
# #plot.ylim([0.0, 1.1*max(mseOob)])
# plot.show()
#
# #plot accuracy
# plot.plot(featureList, accuracy, color='blue')
# plot.xlabel('Number of Features')
# plot.ylabel('Accuracy')
# plot.show()

#plot training and test errors vs number of trees in ensemble
# plot.plot(depthList, auc, color='blue')
# plot.xlabel('Depth of Trees in Ensemble')
# plot.ylabel('Area Under ROC Curve - AUC')
# #plot.ylim([0.0, 1.1*max(mseOob)])
# plot.show()
#
# #plot accuracy
# plot.plot(depthList, accuracy, color='blue')
# plot.xlabel('Depth of Trees in Ensemble')
# plot.ylabel('Accuracy')
# plot.show()

# Plot feature importance
featureImportance = rocksVMinesRFModel.feature_importances_
featureImportance = featureImportance / featureImportance.max() #normalize by max importance
idxSorted = np.argsort(featureImportance) #top 30 [30:60]
barPos = np.arange(idxSorted.shape[0]) + .5
plot.barh(barPos, featureImportance[idxSorted], align='center', color='blue')
plot.yticks(barPos, np.array(columns)[idxSorted])
plot.xlabel('Variable Importance')
plot.subplots_adjust(bottom=0.13, left=0.2)
plot.show()

# Plot best version of ROC curve
fpr, tpr, thresh = roc_curve(yTest, list(prediction[:,1:2]))
ctClass = [i*0.01 for i in range(101)]
plot.plot(fpr, tpr, linewidth=2)
plot.plot(ctClass, ctClass, linestyle=':')
plot.xlabel('False Positive Rate')
plot.ylabel('True Positive Rate')
plot.show()

#pick some threshold values and calc confusion matrix for
#best predictions

#notice that GBM predictions don't fall in range of (0, 1)
#pick threshold values at 25th, 50th and 75th percentiles
idx25 = int(len(thresh) * 0.25)
idx50 = int(len(thresh) * 0.50)
idx75 = int(len(thresh) * 0.75)

#calculate total points, total positives and total negatives
totalPts = len(yTest)
P = sum(yTest)
N = totalPts - P

print('\nConfusion Matrices for Different Threshold Values')

#25th
TP = tpr[idx25] * P; FN = P - TP; FP = fpr[idx25] * N; TN = N - FP
print('')
print('Threshold Value = ', thresh[idx25])
print('TP = ', round(TP/totalPts, 4), int(TP), 'FN = ', round(FN/totalPts, 4), int(FN))
print('FP = ', round(FP/totalPts, 4), int(FP), 'TN = ', round(TN/totalPts, 4), int(TN))
print('Precision = ', TP/(TP+FP))
print('Recall = ', TP/(TP+FN))

#50th
TP = tpr[idx50] * P; FN = P - TP; FP = fpr[idx50] * N; TN = N - FP
print('')
print('Threshold Value = ', thresh[idx50])
print('TP = ', round(TP/totalPts, 4), int(TP), 'FN = ', round(FN/totalPts, 4), int(FN))
print('FP = ', round(FP/totalPts, 4), int(FP), 'TN = ', round(TN/totalPts, 4), int(TN))
print('Precision = ', TP/(TP+FP))
print('Recall = ', TP/(TP+FN))

#75th
TP = tpr[idx75] * P; FN = P - TP; FP = fpr[idx75] * N; TN = N - FP
print('')
print('Threshold Value = ', thresh[idx75])
print('TP = ', round(TP/totalPts, 4), int(TP), 'FN = ', round(FN/totalPts, 4), int(FN))
print('FP = ', round(FP/totalPts, 4), int(FP), 'TN = ', round(TN/totalPts, 4), int(TN))
print('Precision = ', TP/(TP+FP))
print('Recall = ', TP/(TP+FN))

# Printed Output:
#
# AUC
# 0.950304259635
#
# Confusion Matrices for Different Threshold Values
#
# ('Threshold Value = ', 0.76051282051282054)
# ('TP = ', 0.25396825396825395, 'FP = ', 0.0)
# ('FN = ', 0.2857142857142857, 'TN = ', 0.46031746031746029)
#
# ('Threshold Value = ', 0.62461538461538457)
# ('TP = ', 0.46031746031746029, 'FP = ', 0.047619047619047616)
# ('FN = ', 0.079365079365079361, 'TN = ', 0.41269841269841268)
#
# ('Threshold Value = ', 0.46564102564102566)
# ('TP = ', 0.53968253968253965, 'FP = ', 0.22222222222222221)
# ('FN = ', 0.0, 'TN = ', 0.23809523809523808)
