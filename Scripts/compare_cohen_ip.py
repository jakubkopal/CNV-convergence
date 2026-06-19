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
from nilearn import plotting
from nilearn.input_data import NiftiLabelsMasker
from neuromaps import transforms
from neuromaps import nulls
from neuromaps.stats import compare_images

PATH = 'Data/CNV_IP/'
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
category_names = ['1q21.1del', '1q21.1dup', '15q11.2del', '15q11.2dup',
                  '16p11.2del', '16p11.2dup', '22q11.2del', '22q11.2dup']

my_colors = sns.color_palette("tab20", n_colors=len(category_names))
my_colors.reverse()

# Load LDA-derived intermediate phenotypes
tmp = np.load(PATH + 'intermediate_phenotype.npz', allow_pickle=True)
all_models = tmp['all_models']
all_models_avg = tmp['all_models_avg']

# Load LDA classification accuracy
tmp = np.load(PATH + 'model_performance.npz', allow_pickle=True)
mccs = tmp['mccs']
mccs_chance = tmp['mccs_chance']

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

####################################################################
# %% Plot Cohen's d brain maps
####################################################################
# Fsaverage5 surface template
fsaverage = ds.fetch_surf_fsaverage('fsaverage')
# Schaefer atlas
schaefer = ds.fetch_atlas_schaefer_2018(n_rois=400,
                                        yeo_networks=7)
schaefer_rois = np.array(schaefer.labels, dtype=str)
schaefer_atlas = schaefer.maps
masker = NiftiLabelsMasker(labels_img=schaefer_atlas, standardize=False,
                           memory='nilearn_cache')
masker.fit()

img_max = np.max(np.abs(np.array(X_coh)))
for i, cat in enumerate(category_names):
    data = X_coh[i]
    data = np.reshape((data), (-1, 1))
    nifti = masker.inverse_transform(data.T)
    fig = plotting.plot_glass_brain(nifti, threshold=None, colorbar=False,
                                    plot_abs=False, black_bg=False,
                                    cmap='bwr',
                                    symmetric_cbar=True, alpha=1,
                                    display_mode='lzr', vmax=img_max)
    plt.show()

####################################################################
# %% Correlations of Cohen's d maps
#####################################################################


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
            rotated = nulls.alexander_bloch(data2, atlas='fsaverage',
                                            density='10k',
                                            n_perm=1000, seed=1)
            r[j, i], p[j, i] = compare_images(data1, data2, metric='pearsonr',
                                              nulls=rotated)
    p_sig, p_adj = fdrcorrection(p[np.tril_indices(p.shape[0], k=-1)])
    p = np.zeros((n, n))
    p[np.tril_indices(p.shape[0], k=-1)] = p_sig
    return r, p, p_adj


(r, p) = map_correlation(X_coh, len(category_names))

mask = np.triu(np.ones((r.shape[0]-1, r.shape[0]-1)), k=1)

sns.set_theme(style="white", palette='colorblind', font_scale=3)
fig, ax = plt.subplots(1, figsize=(9, 9))
g = sns.heatmap(data=r[1:len(category_names), 0:(len(category_names)-1)],
                mask=mask, cbar=True, cmap='RdBu_r', center=0, linewidths=0.5,
                square=True, annot=False, vmin=-0.55, vmax=0.65,
                cbar_kws={'label': "Pearson\'s r", 'shrink': 0.65,
                          'ticks': [-0.5, -0.3, 0, 0.3, 0.5],
                          'anchor': (0, 0.5), 'pad': -0.10})
for i in range(r.shape[0]):
    for j in range(i+1):
        if p[i, j] == 1:
            if np.abs(r[i, j]) > 0.4:
                ax.text(j + 0.5, i - 0.25, '*', color='w', size=50,
                        ha='center', va='center')
            else:
                ax.text(j + 0.5, i - 0.25, '*', color='k', size=50,
                        ha='center', va='center')
n_cubes = r.shape[1]
plt.axhline(y=n_cubes-1, xmin=0, xmax=1, linestyle='-',
            linewidth=1.5, color='k')
plt.axvline(x=0, ymin=0, ymax=1, linestyle='-', linewidth=1.5,
            color='k')
for i in range(n_cubes):
    plt.axhline(y=i, xmin=i*(1/(n_cubes-1)), xmax=(i+1)*1/(n_cubes-1),
                linestyle='-', linewidth=0.75, color='k')
    plt.axvline(x=n_cubes-1-i, ymin=i*(1/(n_cubes-1)),
                ymax=(i+1)*1/(n_cubes-1),
                linestyle='-', linewidth=0.75, color='k')
g.set_xticklabels(category_names[0:(len(category_names)-1)], rotation=46,
                  horizontalalignment='right')
g.set_yticklabels(category_names[1:len(category_names)], rotation=0)
ax.set_xticklabels(category_names[0:len(category_names)-1], rotation=45,
                   horizontalalignment='right')
ax.set_yticklabels(category_names[1:len(category_names)], rotation=0)
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_title('')
plt.show()

####################################################################
# %% Matthews correlation coefficient of DLA models
#####################################################################
sns.set_theme(style="ticks", palette='colorblind', font_scale=1.8)
fig, ax = plt.subplots(1, figsize=(10, 7))
d = pd.melt(pd.DataFrame(mccs[range(len(category_names)), :].T,
                         columns=category_names))
g = sns.barplot(x='variable', y='value', data=d, palette=my_colors,
                errorbar=None, zorder=1)
for i, cat in enumerate(category_names):
    tmp = np.arange(i-0.38, i+0.38, 0.01)
    plt.plot(tmp, np.ones(len(tmp))*np.percentile(mccs_chance, 97.5,
                                                  axis=1)[i],
             c='k', linewidth=1.5)
    plt.plot(tmp, np.ones(len(tmp))*np.percentile(mccs_chance, 2.5, axis=1)[i],
             c='k', linewidth=1.5)
    plt.fill_between(tmp,
                     np.ones(len(tmp))*np.percentile(mccs_chance, 2.5,
                                                     axis=1)[i],
                     np.ones(len(tmp))*np.percentile(mccs_chance, 97.5,
                                                     axis=1)[i],
                     color='k', alpha=0.2, zorder=2)
plt.axhline(y=0, linestyle='-', linewidth=1, color='k')
g.set_title('')
g.set_xticklabels(category_names, rotation=45, horizontalalignment='right')
g.set_ylabel("Matthews correlation coefficient")
g.set_xlabel('')
ax.yaxis.grid()
plt.ylim(bottom=0)
sns.despine(left=True)
plt.show()

##############################################################################
# %% Plot Cohen's d vs classification accuracy for each CNV
##############################################################################
category_type = ['1q21.1', '15q11.2', '16p11.2', '22q11.2']
(tmp, nmb_cnvs) = np.unique(y, return_counts=True)  # Sample size for each CNV
mean_cohen = []
for i, cat in enumerate(category_names):
    mean_cohen.append(np.mean(np.abs((X_coh[i]))))
d = pd.DataFrame({'coh': mean_cohen,
                  'mcc': np.mean(mccs[range(len(category_names)), :], axis=1),
                  'sizes': nmb_cnvs,
                  'ng': [12, 12, 4, 4, 29, 29, 60, 60],
                  'cat': category_names})
mark_scale = 10
r = np.round(np.corrcoef(d.coh, d.mcc)[0, 1], 2)
sns.set_theme(style="ticks", palette='colorblind', font_scale=1.7)
fig, ax = plt.subplots(1, figsize=(8, 6))
# 1q21 del
tmp = d[d.cat == category_names[0]]
g0 = ax.scatter(x='coh', y='mcc', data=tmp, s=nmb_cnvs[0]*mark_scale,
                c=my_colors[0], linewidth=1, edgecolor='k', marker='o')
ax.annotate(category_names[0], (tmp.coh.values-0.0, tmp.mcc.values-0.0),
            fontsize=20, c='k')
# 1q21 dup
tmp = d[d.cat == category_names[1]]
g1 = ax.scatter(x='coh', y='mcc', data=tmp, s=nmb_cnvs[1]*mark_scale,
                c=my_colors[1], linewidth=1, edgecolor='k', marker='o')
ax.annotate(category_names[1], (tmp.coh.values-0.0, tmp.mcc.values-0.0),
            fontsize=20, c='k')
# 15q11 del
tmp = d[d.cat == category_names[2]]
g2 = ax.scatter(x='coh', y='mcc', data=tmp, s=nmb_cnvs[2]*mark_scale,
                c=my_colors[2], linewidth=1, edgecolor='k', marker='o')
ax.annotate(category_names[2], (tmp.coh.values+0.00, tmp.mcc.values+0.0),
            fontsize=20, c='k')
# 15q11 dup
tmp = d[d.cat == category_names[3]]
g3 = ax.scatter(x='coh', y='mcc', data=tmp, s=nmb_cnvs[3]*mark_scale,
                c=my_colors[3], linewidth=1, edgecolor='k', marker='o')
ax.annotate(category_names[3], (tmp.coh.values, tmp.mcc.values),
            fontsize=20, c='k')
# 16p11 del
tmp = d[d.cat == category_names[4]]
g4 = ax.scatter(x='coh', y='mcc', data=tmp, s=nmb_cnvs[4]*mark_scale,
                c=my_colors[4], linewidth=1, edgecolor='k', marker='o')
ax.annotate(category_names[4], (tmp.coh.values, tmp.mcc.values),
            fontsize=20, c='k')
# 16p11 dup
tmp = d[d.cat == category_names[5]]
g5 = ax.scatter(x='coh', y='mcc', data=tmp, s=nmb_cnvs[5]*mark_scale,
                c=my_colors[5], linewidth=1, edgecolor='k', marker='o')
ax.annotate(category_names[5], (tmp.coh.values-0.0, tmp.mcc.values+0.0),
            fontsize=20, c='k')
# 22q11 del
tmp = d[d.cat == category_names[6]]
g6 = ax.scatter(x='coh', y='mcc', data=tmp, s=nmb_cnvs[6]*mark_scale,
                c=my_colors[6], linewidth=1, edgecolor='k', marker='o')
ax.annotate(category_names[6], (tmp.coh.values+0, tmp.mcc.values+0),
            fontsize=20, c='k')
# 22q11 dup
tmp = d[d.cat == category_names[7]]
g7 = ax.scatter(x='coh', y='mcc', data=tmp, s=nmb_cnvs[7]*mark_scale,
                c=my_colors[7], linewidth=1, edgecolor='k', marker='o')
ax.annotate(category_names[7], (tmp.coh.values, tmp.mcc.values),
            fontsize=20, c='k')
ax.text(0.152, 0.43, "Pearson\'s r = "+str(r), color='black',
        bbox=(dict(edgecolor='black', alpha=1, facecolor='white',
                   mutation_aspect=1.62)))
ax.set_title('Effect size and classification accuracy', fontsize=22)
ax.set_xlabel('Mean absolute Cohen\'s d', fontsize=18)
ax.set_ylabel('Matthews correlation coefficient', fontsize=18)
ax.set(xlim=(0.11, 0.30), ylim=(0.05, 0.78))
ax.xaxis.grid()
ax.yaxis.grid()
plt.show()

##############################################################################
# %% Compare Cohen's d map and intermediate phenotypes
##############################################################################
# Project both maps to brain surfarnce and correlate
# P-value using spin permutation testing
schaefer = ds.fetch_atlas_schaefer_2018(n_rois=400,
                                        yeo_networks=7)
schaefer_atlas = schaefer.maps
masker = NiftiLabelsMasker(labels_img=schaefer_atlas, standardize=False,
                           memory='nilearn_cache')
masker.fit()
r = []
p = []
for i in range(len(category_names)):
    data1 = X_coh[i]
    data1 = np.reshape(data1, (-1, 1))
    data1 = masker.inverse_transform(data1.T)
    data1 = transforms.mni152_to_fsaverage(data1, '10k')
    data2 = all_models_avg[i]
    data2 = np.reshape(data2, (-1, 1))
    data2 = masker.inverse_transform(data2.T)
    data2 = transforms.mni152_to_fsaverage(data2, '10k')
    rotated = nulls.alexander_bloch(data2, atlas='fsaverage', density='10k',
                                    n_perm=10, seed=1)
    tmp_r, tmp_p = compare_images(data1, data2, metric='pearsonr',
                                  nulls=rotated)
    r.append(tmp_r)
    p.append(tmp_p)

r = np.array(r)
d = pd.DataFrame({'r': (r), 'cat': category_names})

# %% Plot Cohen's d map and phenotypes
sns.set_theme(style="ticks", palette='colorblind', font_scale=1.7)
fig, ax = plt.subplots(1, figsize=(9, 5))
g = sns.barplot(x='cat', y='r', data=d, palette=my_colors, errorbar=None)
g.set_title("Intermediate phenotypes resemble Cohen\'s d maps", fontsize=20)
g.set_xticklabels(category_names, rotation=45, horizontalalignment='right')
g.set_ylabel("Pearson\'s r", fontsize=18)
g.set_xlabel('')
ax.yaxis.grid()
for i in range(len(category_names)):
    plt.text(i-0.075, r[i]-0.02, '*', fontsize=28)
plt.ylim(top=np.max(r)+0.07)
sns.despine(left=True)
plt.show()
