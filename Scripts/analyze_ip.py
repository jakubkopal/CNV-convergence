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
from neuromaps import nulls
from neuromaps.stats import compare_images
from math import pi

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
X_full = np.concatenate((X_mut, X_ctrl))
category_names = ['1q21.1del', '1q21.1dup', '15q11.2del', '15q11.2dup',
                  '16p11.2del', '16p11.2dup', '22q11.2del', '22q11.2dup']

my_colors = sns.color_palette("tab20", n_colors=len(category_names))
my_colors.reverse()

n_boot = 100  # nb of bootstraps during LDA model learning
n_iter = 100  # nb of random label permutations

# Load LDA-derived intermediate phenotypes
tmp = np.load(PATH + 'intermediate_phenotype.npz', allow_pickle=True)
all_models = tmp['all_models']
all_models_avg = tmp['all_models_avg']

# Load LDA classification accuracy
tmp = np.load(PATH + 'model_performance.npz', allow_pickle=True)
mccs = tmp['mccs']
mccs_chance = tmp['mccs_chance']

# Supporting network variables
net_dict = ({'Vis': 0, 'SomMot': 1, 'DorsAttn': 2, 'SalVentAttn': 3,
             'Limbic': 4, 'Cont': 5, 'Default': 6})
data = all_models_avg[4, :]
schaefer = ds.fetch_atlas_schaefer_2018(n_rois=400,
                                        yeo_networks=7)
schaefer_rois = np.array(schaefer.labels, dtype=str)
schaefer_atlas = schaefer.maps
schaefer_network = []
for txt in schaefer_rois:
    tmp = txt.split('_')
    schaefer_network.append(tmp[2])
schafer_network_code = pd.Series(schaefer_network).map(net_dict)
schafer_network_code = np.sort(schafer_network_code)
(networks, net_counts) = np.unique(schaefer_network, return_counts=True)
networks[0] = 'FrontPar'

##############################################################################
# %% Bootstrap significance test of LDA coefs
##############################################################################

mask_list = []
p_ttest = np.zeros((len(category_names), X_full.shape[1]))
for i, cat in enumerate(category_names):
    tmp = all_models[i]
    coeff_boot = np.zeros((X_full.shape[1], n_boot))
    for j in range(n_boot):
        coeff_boot[:, j] = tmp[j].coef_
    boot_mean = np.mean(coeff_boot, axis=1)
    boot_ptile_low = np.percentile(coeff_boot, 2.5, axis=1)
    boot_ptile_high = np.percentile(coeff_boot, 97.5, axis=1)
    boot_ptile = (np.sign(boot_ptile_low) == np.sign(boot_ptile_high)) \
        * boot_mean
    mask = (np.sign(boot_ptile_low) == np.sign(boot_ptile_high)) > 0
    mask_list.append(mask)

##############################################################################
# %% Spider plot - proportions of signif LDA coefs
##############################################################################
# Avg number of LDA coefs per CNV
net_avg = []
for i, cat in enumerate(category_names):
    data = mask_list[i]
    d = pd.DataFrame({'Network': schaefer_network, 'Value': np.abs(data)})
    d = d.groupby(['Network']).mean()
    net_avg.append(np.array(d.values)[:, 0])
d = pd.DataFrame((np.array(net_avg)), index=category_names, columns=networks)
d.reset_index(level=0, inplace=True)


# ------- PART 1: Define a function that do a plot for one line of the dataset!
def make_spider(row, title, color):
    # number of variable
    categories = list(d)[1:]
    N = len(categories)
    # What will be the angle of each axis in the plot?
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    # Initialise the spider plot
    sns.set_theme(style="white", palette='colorblind')
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    ax = plt.subplot(1, 1, 1, polar=True)
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    # Draw one axe per variable + add labels labels yet
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels('')
    ax.set_yticks([0.05, 0.1, 0.15, 0.2, 0.25])
    ax.set_yticklabels(["5%", "10%", "15%", '20%', '25%'])
    # Draw ylabels
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 0.20)
    # Ind1
    values = d.loc[row].drop('index').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.6)
    # Add a title
    ax.set_title(title, size=28, color=color, y=1.175, weight='semibold')
    ax.tick_params(axis="x", direction="out", pad=35)
    plt.show()


ax = make_spider(row=0, title=d['index'][0], color=my_colors[0])
# ------- PART 2: Apply the function to all individuals
# Loop through all CNVs
for row in range(0, len(d.index)):
    make_spider(row=row, title=d['index'][row], color=my_colors[row])

##############################################################################
# %% Correlate intermediate phenotypes
##############################################################################


def map_correlation(X, n):
    schaefer = ds.fetch_atlas_schaefer_2018(n_rois=400,
                                            yeo_networks=7)
    schaefer_atlas = schaefer.maps
    masker = NiftiLabelsMasker(labels_img=schaefer_atlas, standardize=False,
                               memory='nilearn_cache')
    masker.fit()
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
            r[j, i], p[j, i] = compare_images(data1, data2,
                                              metric='pearsonr',
                                              nulls=rotated)
    p_sig, p_adj = fdrcorrection(p[np.tril_indices(p.shape[0], k=-1)])
    p = np.zeros((n, n))
    p[np.tril_indices(p.shape[0], k=-1)] = p_sig
    return r, p


(r, p) = map_correlation(np.abs(all_models_avg), len(category_names))
mask = np.triu(np.ones((r.shape[0]-1, r.shape[0]-1)), k=1)

sns.set_theme(style="white", palette='colorblind', font_scale=2)
fig, ax = plt.subplots(1, figsize=(9, 9))
g = sns.heatmap(data=r[1:len(category_names), 0:(len(category_names)-1)],
                mask=mask, cbar=True, cmap='RdBu_r', center=0, linewidths=0.5,
                square=True, annot=False, vmin=-0.55, vmax=0.65,
                cbar_kws={'label': "Pearson\'s r", 'shrink': 0.75,
                          'ticks': [-0.5, -0.3, 0, 0.3, 0.5],
                          'anchor': (0, 0.9), 'pad': -0.125})
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
                linestyle='-', linewidth=1.5, color='k')
    plt.axvline(x=n_cubes-1-i, ymin=i*(1/(n_cubes-1)),
                ymax=(i+1)*1/(n_cubes-1),
                linestyle='-', linewidth=1.5, color='k')
g.set_xticklabels(category_names[0:(len(category_names)-1)], rotation=46,
                  horizontalalignment='right')
g.set_yticklabels(category_names[1:len(category_names)], rotation=0)
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_title('')
plt.show()

####################################################################
# %% Compared networks in terms of number of LDA coefficients
####################################################################
# Compute mean number of LDA coefs per network
net_avg = []
for i, cat in enumerate(category_names):
    data = mask_list[i]
    d = pd.DataFrame({'Network': schaefer_network, 'Value': np.abs(data)})
    d = d.groupby(['Network']).mean()
    net_avg.append(np.array(d.values)[:, 0])

net_avg = np.array(net_avg)
net_avg_mean = np.mean(net_avg, axis=0)
idx_sort = np.argsort(net_avg)[::-1]

networks = np.array(['Frontoparietal', 'Default mode', 'Dorsal attention',
                     'Limbic', 'Salience \n ventral attention',  'Somatomotor',
                     'Visual'])
d = pd.DataFrame(net_avg[:, idx_sort]*100, index=category_names,
                 columns=networks[idx_sort])
d_alt = pd.DataFrame(net_avg[:, idx_sort]*100, index=category_names)
t = pd.melt(d_alt, ignore_index=False)
t = t.reset_index(drop=False)
PROPS = {
    'boxprops': {'facecolor': 'none', 'edgecolor': 'grey'},
    'medianprops': {'color': 'black'},
    'whiskerprops': {'color': 'grey'},
    'capprops': {'color': 'grey'}
}
MARKERS = ['_', '+']
sns.set_theme(style="white", palette='colorblind', font_scale=1.8)
fig, ax = plt.subplots(1, figsize=(6, 10))
g = sns.boxplot(x='value', y='variable', data=t, orient='horizontal',
                color='w', ax=ax, fliersize=0, zorder=0, width=0.7, **PROPS)
for i in range(len(d.columns)):
    tmp = pd.DataFrame(np.array([d.iloc[:, i].values,
                                 np.arange(0,
                                           len(category_names))*0.05+i-0.2]).T,
                       columns=['index', 'r'], index=category_names)
    tmp = tmp.reset_index(drop=False)
    g = sns.scatterplot(x='index', y='r', hue='level_0', data=tmp, ax=ax,
                        palette=my_colors, linewidth=5,
                        legend=False, s=200, alpha=1, zorder=10*i,
                        style=np.tile((0, 1), 4), markers=MARKERS)
for i in range(len(d.columns)):
    plt.axhline(y=i+0.5, linestyle='--', linewidth=0.5, color='k')
ax.set_title('')
ax.set_ylabel('')
ax.set_xlabel('Proportion of significant ROIs [%]')
ax.set_yticks(np.arange(0, len(d.columns)))
ax.set_yticklabels(networks[idx_sort], fontsize=22)
ax.xaxis.grid()
plt.xlim(left=0)
sns.despine(left=True)
plt.show()
