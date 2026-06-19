#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute and plot Cohen's d brain maps. Compare them with LDA coefficients'
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import fdrcorrection
from nilearn import datasets as ds
from nilearn.input_data import NiftiLabelsMasker
from neuromaps import transforms
from neuromaps.stats import compare_images
from scipy.stats import pearsonr
from matplotlib.lines import Line2D

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

my_colors = sns.color_palette("tab20", n_colors=len(category_names))
my_colors.reverse()

n_boot = 100  # nb of bootstraps during LDA model learning
n_iter = 100  # nb of random label permutations

# Load LDA-derived intermediate phenotypes
tmp = np.load('Data/CNV_IP/intermediate_phenotype.npz', allow_pickle=True)
all_models = tmp['all_models']
all_models_avg = tmp['all_models_avg']

# Load LDA classification accuracy
phewas_cnv = np.load('Data/PheWAS/phewas_outcomes.npz')

##############################################################################
# %% Correlate PheWAS outcomes
##############################################################################


def map_pearson(X, n):
    r = np.zeros((n, n))
    p = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            r[i, j], p[i, j] = pearsonr(X[i], X[j])
    p_sig, p_adj = fdrcorrection(p[np.tril_indices(p.shape[0], k=-1)])
    p = np.zeros((n, n))
    p[np.tril_indices(p.shape[0], k=-1)] = p_sig
    return r, p


(r, p) = map_pearson(np.array(phewas_cnv)[0:len(category_names), :],
                     len(category_names))
mask = np.triu(np.ones((r.shape[0]-1, r.shape[0]-1)), k=1)

sns.set_theme(style="white", palette='colorblind', font_scale=2)
fig, ax = plt.subplots(1, 1, figsize=(9, 9))
g = sns.heatmap(data=r[1:len(category_names), 0:(len(category_names)-1)],
                mask=mask, cbar=True, cmap='RdBu_r', center=0, linewidths=0.1,
                square=True, annot=False,
                cbar_kws={'label': "Pearson\'s r", 'shrink': 0.75,
                          'ticks': [-0.8, -0.4, 0, 0.4, 0.8],
                          'anchor': (0, 0.9), 'pad': -0.1})
for i in range(r.shape[0]):
    for j in range(i+1):
        if p[i, j] == 1:
            if np.abs(r[i, j]) > 0.4:
                ax.text(j + 0.5, i - 0.3, '*', color='w', size=50,
                        ha='center', va='center')
            else:
                ax.text(j + 0.5, i - 0.3, '*', color='k', size=50,
                        ha='center', va='center')
n_cubes = r.shape[1]
plt.axhline(y=n_cubes-1, xmin=0, xmax=1, linestyle='-',
            linewidth=1.5, color='k')
plt.axvline(x=0, ymin=0, ymax=1, linestyle='-', linewidth=1.5,
            color='k')
for i in range(n_cubes):
    plt.axhline(y=i, xmin=i*(1/(n_cubes-1)), xmax=(i+1)*1/(n_cubes-1),
                linestyle='-', linewidth=1.5, color='k')
    plt.axvline(x=n_cubes-1-i, ymin=i*(1/(n_cubes-1)),
                ymax=(i+1)*1/(n_cubes-1),
                linestyle='-', linewidth=1.5, color='k')
plt.axhline(y=0, xmin=0, xmax=1/(n_cubes-1), linestyle='-', linewidth=1.5,
            color='k')
plt.axvline(x=n_cubes-1, ymin=0, ymax=1/(n_cubes-1), linestyle='-',
            linewidth=1.5, color='k')
ax.set_title('')
ax.set_xticklabels(category_names[0:len(category_names)-1], rotation=45,
                   horizontalalignment='right')
ax.set_yticklabels(category_names[1:len(category_names)], rotation=0)
ax.set_ylabel('')
ax.set_xlabel('')
plt.show()

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

##############################################################################
# %% Concordance plot
##############################################################################


def map_correlation(X, n):
    schaefer = ds.fetch_atlas_schaefer_2018(n_rois=400,
                                            yeo_networks=7)
    schaefer_atlas = schaefer.maps
    masker = NiftiLabelsMasker(labels_img=schaefer_atlas, standardize=False,
                               memory='nilearn_cache')
    masker.fit()

    schaefer = ds.fetch_atlas_schaefer_2018(n_rois=400,
                                            yeo_networks=7)
    r = np.zeros((n, n))
    p = np.zeros((n, n))
    for i in range(n-1):
        print('Row:' + str(i))
        data1 = X[i]
        data1 = np.reshape(data1, (-1, 1))
        data1 = masker.inverse_transform(data1.T)
        data1 = transforms.mni152_to_fsaverage(data1, '10k')
        for j in np.arange(i+1, n):
            data2 = X[j]
            data2 = np.reshape(data2, (-1, 1))
            data2 = masker.inverse_transform(data2.T)
            data2 = transforms.mni152_to_fsaverage(data2, '10k')
            r[j, i] = compare_images(data1, data2, metric='pearsonr')
    return r, p


def ccc(x, y):
    ''' Concordance Correlation Coefficient'''
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rhoc = 2*sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)
    return rhoc


# Cohen's brain map similarity
r_coh, p = map_correlation(X_coh, len(category_names))
# Phenotypic similarity
r_phen = np.corrcoef(phewas_cnv)
# Intermediate phenotype similarity
(r_ip, p) = map_correlation(all_models_avg, len(category_names))


d = pd.DataFrame({'coh': np.abs(r_coh[np.tril_indices(len(category_names),
                                                      k=-1)]),
                  'phen': np.abs(r_phen[np.tril_indices(len(category_names),
                                                        k=-1)])})
idx_list = []
for i in range(len(d)):
    if d.iloc[i, 0] > d.iloc[i, 1]:
        idx_list.append(i)
id1, id2 = np.tril_indices(len(category_names), k=-1)
r = np.round(ccc(d.coh, d.phen), 2)

sns.set_theme(style="ticks", palette='colorblind', font_scale=1.5)
fig, ax = plt.subplots(1, figsize=(8, 8))
for i in range(len(id1)):
    line = Line2D([d.coh[i]], [d.phen[i]], linewidth=0, marker='o',
                  markersize=20, markeredgecolor='k',
                  markerfacecolor=my_colors[id1[i]],
                  markerfacecoloralt=my_colors[id2[i]],
                  fillstyle='right')
    ax.add_line(line)
ax.text(0.345, 0.84, "Lin\'s concordance = "+str(r), color='black',
        fontsize=22,
        bbox=(dict(edgecolor='black', alpha=1, facecolor='white',
                   mutation_aspect=1.62)))
ax.set_title('')
ax.set_ylabel('Phenotypic similarity ', fontsize=20)
ax.set_xlabel('Volumetric similarity', fontsize=20)
ax.set(xlim=(-0.0, 0.9), ylim=(-0.0, 0.9))
ax.xaxis.grid()
ax.yaxis.grid()
ax.set_aspect('equal', adjustable='box')
ax.plot([-1, 1], [-1, 1], ls="--", c=".3")
plt.show()

##############################################################################
# %% Lineplot Converging CNVs
##############################################################################
tmp = np.array([np.abs(r_coh[np.tril_indices(len(category_names), k=-1)]),
                np.abs(r_ip[np.tril_indices(len(category_names), k=-1)]),
                np.abs(r_phen[np.tril_indices(len(category_names), k=-1)])]).T
d = pd.melt(pd.DataFrame(tmp))
d['pair'] = np.tile(np.arange(0, tmp.shape[0]), tmp.shape[1])

sns.set_theme(style="ticks", palette='colorblind', font_scale=1.5)
fig, ax = plt.subplots(1, figsize=(9, 6))
for i in range(tmp.shape[0]):
    if d[d.pair == i].value.values[0] < d[d.pair == i].value.values[2]:
        g = sns.lineplot(x='variable', y='value', data=d[d.pair == i],
                         color=sns.color_palette('RdBu', 8)[0],
                         legend=False)
    else:
        g = sns.lineplot(x='variable', y='value', data=d[d.pair == i],
                         color=sns.color_palette('RdBu', 8)[-1],
                         legend=False)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["Volumetric\nCohen\'s d",
                    'Intermediate\nphenotype', 'PheWAS'],
                   rotation=30, horizontalalignment='right')
ax.set_ylabel("Pearson\'s r")
ax.set_xlabel('Association strength')
ax.set_xlim([0, 2])
ax.set_ylim([0, 1])
ax.yaxis.grid()
ax.xaxis.grid()
plt.show()
