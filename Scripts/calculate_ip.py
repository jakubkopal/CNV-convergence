#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract intermediate phenotype from clinical cohort
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import shuffle
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
from tqdm import tqdm

scale = StandardScaler()

#####################################################################
# %% Prepare data
#####################################################################
# Vector containing CNV codes for each subject
# 1q21.1 del - 1
# 1q21.1 dup - 1
# 15q11.2 del - 1
# 15q11.2 dup - 1
# 16p11.2 del - 1
# 16p11.2 dup - 1
# 22q11.2 del - 1
# 22q11.2 dup - 1
y = np.load('')  # vector containing CNV codes for each subject
X_mut = np.load('')  # matrix containing regional volumes for each CNV carrier
X_ctrl = np.load('')  # matrix containing regional volumes for each control
X_full = np.concatenate((X_mut, X_ctrl))
category_names = ['1q21.1del', '1q21.1dup', '15q11.2del', '15q11.2dup',
                  '16p11.2del', '16p11.2dup', '22q11.2del', '22q11.2dup']

n_boot = 100  # nb of bootstraps during LDA model learning
n_iter = 100  # nb of random label permutations

#####################################################################
# %%  Calculate Cohen's d using brain volumes
####################################################################


def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    res = (np.mean(x) - np.mean(y)) / \
        np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 +
                 (ny-1)*np.std(y, ddof=1) ** 2) / dof)
    return (res)


# Cohens d
X_coh = []
for i, cat in enumerate(category_names):
    X_cnv = X_mut[y == i, :]
    X_star = np.concatenate((X_cnv, X_ctrl))
    scale.fit(X_star)
    X_ctrl_ss = scale.transform(X_ctrl)
    X_cnv_ss = scale.transform(X_cnv)
    tmp_y = y[y == i]

    y_star = np.concatenate((tmp_y, np.ones(X_ctrl.shape[0])*10))
    y_star = (y_star < 10)*1
    cnv_coh = np.zeros(X_cnv.shape[1])
    for j in range(X_cnv.shape[1]):
        cnv_coh[j] = cohen_d(X_cnv_ss[:, j], X_ctrl_ss[:, j])
    X_coh.append(cnv_coh)

#####################################################################
# Classification
####################################################################


def create_traintest(X, y, n_ctrl, n_mut, boot):
    X_ctrl = X[y == 0]
    X_mut = X[y == 1]
    id_ctrl = np.arange(0, n_ctrl)
    id_mut = np.arange(0, n_mut)
    id_ctrl_train = resample(id_ctrl, replace=False, n_samples=n_mut,
                             random_state=boot)
    id_mut_train = resample(id_mut, replace=True, n_samples=n_mut,
                            random_state=boot)
    id_ctrl_test = np.setdiff1d(id_ctrl, id_ctrl_train)
    id_mut_test = np.setdiff1d(id_mut, id_mut_train)
    id_ctrl_test = np.random.choice(id_ctrl_test, size=len(id_mut_test),
                                    replace=False)  # same number of train
    X_train = np.concatenate((X_ctrl[id_ctrl_train, :],
                              X_mut[id_mut_train, :]))
    X_test = np.concatenate((X_ctrl[id_ctrl_test, :], X_mut[id_mut_test, :]))
    y_train = np.concatenate((np.zeros(n_mut), np.ones(n_mut)))
    y_test = np.concatenate((np.zeros(id_ctrl_test.shape[0]),
                             np.ones(id_mut_test.shape[0])))
    return X_train, X_test, y_train, y_test, id_mut_test


def optimize_hyperparameter(X_train, y_train, cat_id):
    model = LinearDiscriminantAnalysis(solver='eigen')
    param = {"shrinkage": np.arange(0.01, 0.91, 0.1)}
    loo = LeaveOneOut()
    search = GridSearchCV(model, param, scoring='accuracy', cv=loo,
                          n_jobs=-1, refit=True)
    # execute search
    result = search.fit(X_train, y_train)
    # get the best performing model fit on the whole training set
    best_model = result.best_estimator_
    best_model.fit(X_train, y_train)
    # re-order discriminant function
    if np.corrcoef(best_model.scalings_[:, 0], X_coh[cat_id, :])[0, 1] < 0:
        best_model.scalings_ = -best_model.scalings_
    return best_model


def boot_performance(X_test, y_test, model):
    y_predict = model.predict(X_test)
    # Performance
    acc = balanced_accuracy_score(y_test, y_predict, adjusted=False)
    f1 = f1_score(y_test, y_predict)
    mcc = matthews_corrcoef(y_test, y_predict)
    return acc, f1, mcc


def boot_main(cat_name, cat_id, X, y, boot):
    n_ctrl = np.sum(y == 0)  # nb of controls
    n_mut = np.sum(y == 1)  # nb of subjects
    X_train, X_test, y_train, y_test, id_mut_test = create_traintest(X, y,
                                                                     n_ctrl,
                                                                     n_mut,
                                                                     boot)
    model = optimize_hyperparameter(X_train, y_train, cat_id)
    acc, f1, mcc = boot_performance(X_test, y_test, model)
    return model,  acc, f1, mcc


def performance_chance(X, y, model, n_iter):
    acc = np.zeros(n_iter)
    f1 = np.zeros(n_iter)
    mcc = np.zeros(n_iter)
    # Label permutation
    for i in range(n_iter):
        rand_y = shuffle(y, random_state=i)
        y_predict = (np.matmul(X, model) > 0)*1
        # Chance
        acc[i] = balanced_accuracy_score(rand_y, y_predict, adjusted=False)
        f1[i] = f1_score(rand_y, y_predict)
        mcc[i] = matthews_corrcoef(rand_y, y_predict)
    return acc, f1, mcc


###############################################################################
# Initialize variables
accs = np.zeros([len(category_names), n_boot])
mccs = np.zeros([len(category_names), n_boot])
f1s = np.zeros([len(category_names), n_boot])
accs_chance = np.zeros([len(category_names), n_iter])
f1s_chance = np.zeros([len(category_names), n_iter])
mccs_chance = np.zeros([len(category_names), n_iter])
all_models = []
all_models_avg = np.zeros([len(category_names), X_full.shape[1]])
# Learn bagged LDA models for each CNV
for i, category in enumerate(category_names):
    print(category)
    tmp_X = X_mut[y == i, :]
    X_star = np.concatenate((tmp_X, X_ctrl))
    X_star_ss = scale.fit_transform(X_star)
    tmp_y = y[y == i]

    y_star = np.concatenate((tmp_y, np.ones(X_ctrl.shape[0])*10))
    y_star = (y_star < 10)*1
    models = []
    for j in tqdm(np.arange(0, n_boot)):
        model, accs[i, j], f1s[i, j], mccs[i, j] = boot_main(category, i,
                                                             X_star_ss,
                                                             y_star, j)
        models.append(model)

    average_model = np.average(np.array([model.scalings_[:, 0] for model in models]), axis=0)
    accs_chance[i, :], f1s_chance[i, :], mccs_chance[i, :] = performance_chance(X_star_ss,
                                                                                y_star,
                                                                                average_model,
                                                                                n_iter)
    all_models.append(models)
    all_models_avg[i, :] = average_model