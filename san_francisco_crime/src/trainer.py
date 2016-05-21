#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sklearn.svm

import preprocess

data = preprocess.SanFranciscoCrimeData()

# get features for train
x_day = np.asarray([data.get_day_of_week_vector(d) for d in data.train_df['DayOfWeek']])
x_pd = np.asarray([data.get_pd_district_vector(d) for d in data.train_df['PdDistrict']])
x_longtitude = np.asarray([[x_] for x_ in data.train_df['X']])
x_latitude = np.asarray([[y_] for y_ in data.train_df['Y']])
X = np.concatenate((x_day, x_pd, x_longtitude, x_latitude), axis=1)
y = data.get_category_matrix()

# get features for test
t_day = np.asarray([data.get_day_of_week_vector(d) for d in data.test_df['DayOfWeek']])
t_pd = np.asarray([data.get_pd_district_vector(d) for d in data.test_df['PdDistrict']])
t_longtitude = np.asarray([x_ for x_ in data.test_df['X']])
t_latitude = np.asarray([y_ for y_ in data.test_df['Y']])
T = np.concatenate((t_day, t_pd, t_longtitude, t_latitude), axis=1)

# model : SVM
classifier = sklearn.svm.SVC(C=1.0,
                             kernel='rbf')

classifier.fit(X, y)
score = classifier.predict(T)

save_name = 'SVM_result.csv'
np.savetxt(save_name, score, delimiter=',')
