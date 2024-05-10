# -*- coding:utf-8 -*-
import pandas as pd
import MySQLdb
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from color import color


def read_table(cur, sql_order):
    try:
        cur.execute(sql_order)
        data = cur.fetchall()
        frame = pd.DataFrame(list(data))
    except:
        frame = pd.DataFrame()
    return frame

con = MySQLdb.connect('localhost', 'root', 'root', 'psbc', charset='utf8')
cur = con.cursor()

sql = 'select custr_nbr, use_rate, val, el from a00_ccb_tb6_bk;'
frame = read_table(cur, sql)
frame.columns = ['custr_nbr', 'use_rate','val','el']
frame = frame.set_index(['custr_nbr'])

plt.pcolor(frame.iloc[:,0:3].corr())
plt.show()
#           use_rate       val        el
# use_rate  1.000000  0.282179  0.233156
# val       0.282179  1.000000  0.237868
# el        0.233156  0.237868  1.000000


X_train = np.array(frame[['use_rate','val','el']].values)
#X_train = np.array(frame.iloc[:,0:3].values)

X_scale = scale(X_train)
#min_max_scaler = MinMaxScaler()
#X_scale = min_max_scaler.fit_transform(X_train)
#X_scale = X_train    ###不标准化


def print_mean(frame, n, col_list):
    for i in range(n):
        frame = frame[frame.label==i]
        ct = len(frame)
        print("cluster"+str(i)+": "+str(ct))
        mean_list = []
        for j in range(len(col_list)):
            mean_value = round(np.mean(frame[col_list[j]]),4)
            mean_list.append(mean_value)
        print(mean_list)


def find_cent(df, frame, centroids):
    for j in range(len(centroids)):
        tmp= abs(df- centroids[j])
        lst= []
        for i in range(tmp.shape[1]):
            idx = (tmp[i]).tolist().index(min(tmp[i]))
            lst.append(frame.ix[idx][i])
        print("cluster"+str(j)+": "+str(lst)+" "+str(len(frame[frame.label==j])))


def draw_scatter(n, X_scale, frame):
    tsne = TSNE()
    tsne.fit_transform(X_scale)
    X_tsne = pd.DataFrame(tsne.embedding_, index=frame.index)
    ax = plt.subplot(111)
    for i in range(n):
        d = X_tsne[frame['label']==i]
        ax.scatter(d[0], d[1], c=color.names[i])
    plt.show()


def draw_scatter_3d(frame, X_scale, n):
    X_tsne = pd.DataFrame(X_scale, index=frame.index)
    #x, y, z = X_tsne[0], X_tsne[1], X_tsne[2]
    #x, y, z = X_tsne[1], X_tsne[2], X_tsne[0]
    x, y, z = X_tsne[2], X_tsne[0], X_tsne[1]
    #x, y, z = X_tsne[0], X_tsne[2], X_tsne[1]
    #x, y, z = X_tsne[2], X_tsne[1], X_tsne[0]
    #x, y, z = X_tsne[1], X_tsne[0], X_tsne[2]
    ax = plt.subplot(111, projection='3d')
    for i in range(n):
        ax.scatter(x[frame.label==i], y[frame.label==i], z[frame.label==i], c=color.names[i])
    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()


def kmeans(frame, X_scale, n):
    estimator = KMeans(n_clusters=n)
    estimator.fit(X_scale)
    label_pred = estimator.labels_
    centroids = estimator.cluster_centers_
    inertia = estimator.inertia_
    frame['label'] = label_pred
    print("mean")
    print_mean(frame, n)
    print("cent")
    df = pd.DataFrame(X_scale, index=frame.index)
    find_cent(df, frame, centroids)
    #draw_scatter(frame, X_scale, n)
    draw_scatter_3d(frame, X_scale, n)


kmeans(frame, X_scale, 7)
# for i in range(len(frame)):
#     sql = "update a00_ccb_tb6_bk set label7 = '%s' where custr_nbr = '%s';" % (frame.label[i], frame.index[i])
#     try:
#         cur.execute(sql)
#         # con.commit()
#         print "1 row updated."
#     except:
#         con.rollback()
# con.commit()
cur.close()
con.close()


