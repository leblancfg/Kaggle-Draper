import time
start_time = time.time()

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
from PIL import Image
from PIL import ImageFilter
import multiprocessing
import random; random.seed(2016);
import cv2
import re
import os, glob

import warnings
import cv2
warnings.filterwarnings('ignore')
import multiprocessing
from sklearn import ensemble
from sklearn import pipeline, grid_search
from sklearn.metrics import label_ranking_average_precision_score as lraps

def image_features(path, tt, group, pic_no):
    im = cv2.imread(path)
    me_ = cv2.mean(im)
    s = [path, tt, group, pic_no, im.mean(), me_[2], me_[1], me_[0]]
    f = open("data.csv", "a")
    f.write((',').join(map(str, s)) + '\n')
    f.close()
    return


f = open("data.csv", "w");
col = ['path', 'tt', 'group', 'pic_no', 'individual_im_mean', 'rm', 'bm', 'gm']
f.write((',').join(map(str, col)) + '\n')
f.close()

sample_sub = pd.read_csv('sample_submission.csv')
train_files = pd.DataFrame([[f,f.split("/")[3].split(".")[0].split("_")[0],f.split("/")[3].split(".")[0].split("_")[1]] for f in glob.glob("../input/train_sm/*.jpeg")])
train_files.columns = ['path', 'group', 'pic_no']
test_files = pd.DataFrame([[f,f.split("/")[3].split(".")[0].split("_")[0],f.split("/")[3].split(".")[0].split("_")[1]] for f in glob.glob("../input/test_sm/*.jpeg")])
test_files.columns = ['path', 'group', 'pic_no']
print(len(train_files),len(test_files),len(sample_sub))
train_images = train_files[train_files["group"]=='set107']
train_images = train_images.sort_values(by=["pic_no"], ascending=[1]).reset_index(drop=True)


train_images = train_files[train_files["group"]=='set4']
train_images = train_images.sort_values(by=["pic_no"], ascending=[1]).reset_index(drop=True)
img = cv2.imread(train_images.path[0])
cv2.imwrite('panoramic2.jpeg',img)
plt.rcParams['figure.figsize'] = (12.0, 12.0)
for i in range(1,5):
    img = im_stitcher(train_images.path[i], 'panoramic2.jpeg', 0.5, False)
    cv2.imwrite('panoramic2.jpeg',img)
img[np.where((img < [20,20,20]).all(axis = 2))] = [255,255,255]
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.axis('off')

if __name__ == '__main__':
    cpu = multiprocessing.cpu_count();
    print (cpu);

    j = []
    for s_ in range(0, len(train_files), cpu):  # train
        for i in range(cpu):
            i_ = s_ + i
            if (i_) < len(train_files):
                if i_ % 100 == 0:
                    print("train ", i_)
                filename = train_files.path[i_]
                p = multiprocessing.Process(target=image_features, args=(
                filename, 'train', train_files["group"][i_], train_files["pic_no"][i_],))
                j.append(p)
                p.start()
    j = []
    for s_ in range(0, len(test_files), cpu):  # test
        for i in range(cpu):
            i_ = s_ + i
            if (i_) < len(test_files):
                if i_ % 100 == 0:
                    print("test ", i_)
                filename = test_files.path[i_]
                p = multiprocessing.Process(target=image_features,
                                            args=(filename, 'test', test_files["group"][i_], test_files["pic_no"][i_],))
                j.append(p)
                p.start()

    while len(j) > 0:  # end all jobs
        j = [x for x in j if x.is_alive()]
        time.sleep(1)
    df_all = pd.read_csv('data.csv', index_col=None)
    df_all = df_all.reset_index(drop=True)
    df_all['group_min_im_mean'] = df_all["group"].map(
        lambda x: df_all[df_all['group'] == x]['individual_im_mean'].min())
    df_all['group_max_im_mean'] = df_all["group"].map(
        lambda x: df_all[df_all['group'] == x]['individual_im_mean'].max())
    df_all['group_mean'] = df_all["group"].map(lambda x: df_all[df_all['group'] == x]['individual_im_mean'].mean())
    df_all['a'] = df_all['individual_im_mean'] - df_all['group_min_im_mean']
    df_all['b'] = df_all['group_max_im_mean'] - df_all['individual_im_mean']
    df_all['c'] = df_all['group_mean'] - df_all['individual_im_mean']

    # red
    df_all['group_min_im_mean_r'] = df_all["group"].map(lambda x: df_all[df_all['group'] == x]['rm'].min())
    df_all['group_max_im_mean_r'] = df_all["group"].map(lambda x: df_all[df_all['group'] == x]['rm'].max())
    df_all['group_mean_r'] = df_all["group"].map(lambda x: df_all[df_all['group'] == x]['rm'].mean())
    df_all['a_r'] = df_all['rm'] - df_all['group_min_im_mean_r']
    df_all['b_r'] = df_all['group_max_im_mean_r'] - df_all['rm']
    # df_all['c_r'] = df_all['group_mean_r'] - df_all['rm']

    # green
    df_all['group_min_im_mean_g'] = df_all["group"].map(lambda x: df_all[df_all['group'] == x]['gm'].min())
    df_all['group_max_im_mean_g'] = df_all["group"].map(lambda x: df_all[df_all['group'] == x]['gm'].max())
    df_all['group_mean_g'] = df_all["group"].map(lambda x: df_all[df_all['group'] == x]['gm'].mean())
    df_all['a_g'] = df_all['gm'] - df_all['group_min_im_mean_g']
    df_all['b_g'] = df_all['group_max_im_mean_g'] - df_all['gm']
    # df_all['c_g'] = df_all['group_mean_g'] - df_all['gm']

    # blue
    df_all['group_min_im_mean_b'] = df_all["group"].map(lambda x: df_all[df_all['group'] == x]['bm'].min())
    df_all['group_max_im_mean_b'] = df_all["group"].map(lambda x: df_all[df_all['group'] == x]['bm'].max())
    df_all['group_mean_b'] = df_all["group"].map(lambda x: df_all[df_all['group'] == x]['bm'].mean())
    df_all['a_b'] = df_all['bm'] - df_all['group_min_im_mean_b']
    df_all['b_b'] = df_all['group_max_im_mean_b'] - df_all['bm']
    # df_all['c_b'] = df_all['group_mean_b'] - df_all['bm']

    df_all['setId'] = df_all["group"].map(lambda x: x.replace('set', ''))
    df_all.to_csv('data.csv')
    print("Features Ready: ", round(((time.time() - start_time) / 60), 2))

    X_train = df_all[df_all['tt'] == 'train']
    X_train = X_train.sort_values(by=['setId', 'pic_no'], ascending=[1, 1])
    X_train = X_train.reset_index(drop=True)
    y_train = X_train["pic_no"].values
    X_train = X_train.drop(
        ['path', 'tt', 'group', 'pic_no', 'setId', 'individual_im_mean', 'group_min_im_mean', 'group_max_im_mean',
         'group_mean', 'rm', 'group_min_im_mean_r', 'group_max_im_mean_r', 'group_mean_r', 'gm', 'group_min_im_mean_g',
         'group_max_im_mean_g', 'group_mean_g', 'bm', 'group_min_im_mean_b', 'group_max_im_mean_b', 'group_mean_b'],
        axis=1)
    X_test = df_all[df_all['tt'] == 'test']
    X_test = X_test.sort_values(by=['setId', 'pic_no'], ascending=[1, 1])
    # X_test.fillna(0, inplace=True)
    X_test = X_test.reset_index(drop=True)
    id_test = X_test[["setId", "pic_no"]]  # .values
    X_test = X_test.drop(
        ['path', 'tt', 'group', 'pic_no', 'setId', 'individual_im_mean', 'group_min_im_mean', 'group_max_im_mean',
         'group_mean', 'rm', 'group_min_im_mean_r', 'group_max_im_mean_r', 'group_mean_r', 'gm', 'group_min_im_mean_g',
         'group_max_im_mean_g', 'group_mean_g', 'bm', 'group_min_im_mean_b', 'group_max_im_mean_b', 'group_mean_b'],
        axis=1)




    # rfr = ensemble.RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=2016, verbose=0)
    # param_grid = {'max_depth': [6], 'max_features': [1.0]}
    # model = grid_search.GridSearchCV(estimator=rfr, param_grid=param_grid, n_jobs=-1, cv=2, verbose=0)
    # model.fit(X_train, y_train)
    # print("Best parameters found by grid search:")
    # print(model.best_params_)
    # print("Best CV score:", model.best_score_)
    # y_pred = model.predict_proba(X_test)
    # # y_pred = model.predict(X_test)
    # df = pd.concat((pd.DataFrame(id_test), pd.DataFrame(y_pred)), axis=1)
    # df.columns = ['setId', 'pic_no', 'day1', 'day2', 'day3', 'day4', 'day5']
    # # df.to_csv('submission2.csv',index=False)
    # f = open('submission.csv', 'w')
    # f.write('setId,day\n')
    # setID = df.setId.unique()
    # for i in setID:
    #     a = []
    #     df1 = df[df['setId'] == str(i)].reset_index(drop=True)
    #     for j in range(1, 6):
    #         df1 = df1.sort_values(by=['day' + str(j)], ascending=[0]).reset_index(drop=True)
    #         # print(df1)
    #         a.append(df1.pic_no[0])
    #         df1 = df1[1:]
    #     f.write(str(i) + "," + " ".join(map(str, a)) + "\n")
    #     # break
    # f.close()
    # print("Ready to submit: ", round(((time.time() - start_time) / 60), 2))