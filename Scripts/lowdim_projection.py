#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot PCA and LDA two-dimensional projection of regional volumes
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from matplotlib.legend_handler import HandlerTuple
import matplotlib.ticker as tick

PATH = 'Data/projection'
scale = StandardScaler()
pca = PCA(n_components=2)

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

####################################################################
# %% Low-dimensional projection
#####################################################################
X_ctrl_ss = scale.fit_transform(X_ctrl)
X_mut_ss = scale.transform(X_mut)
X = X_mut_ss

# LDA
model = LinearDiscriminantAnalysis(solver='eigen')
model.fit(X, y)
coeffs = model.coef_.T
X_LDA = model.transform(X)
scales = model.scalings_

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_mut_ss)

# %% Plot Two-dimensional PCA
category_type = ['1q21.1', '15q11.2', '16p11.2', '22q11.2']
d = pd.DataFrame({'C1': X_pca[:, 0], 'C2': X_pca[:, 1],
                  'Mutation': y})

sns.set_theme(style="ticks", palette='colorblind', font_scale=1.75)
fig, ax = plt.subplots(1, figsize=(9, 9))
hb = ax.hexbin(pca.transform(X_ctrl_ss)[:, 0],
               pca.transform(X_ctrl_ss)[:, 1],
               cmap='Greys', gridsize=(10, 5))
cb = fig.colorbar(hb, ax=ax, shrink=0.75,
                  format=tick.FormatStrFormatter('%.0f'))
cb.set_label('Number of controls', size=20)
tmp = d[d.Mutation == 2]
g3 = ax.scatter(x='C1', y='C2', data=tmp, s=90, c=my_colors[2], linewidth=0.5,
                edgecolor='k', marker='^', alpha=1)
tmp = d[d.Mutation == 3]
g4 = ax.scatter(x='C1', y='C2', data=tmp, s=90, c=my_colors[3], linewidth=0.5,
                edgecolor='k', marker='^', alpha=1)
tmp = d[d.Mutation == 4]
g5 = ax.scatter(x='C1', y='C2', data=tmp, s=90, c=my_colors[4], linewidth=0.5,
                edgecolor='k', marker='P', alpha=1)
tmp = d[d.Mutation == 5]
g6 = ax.scatter(x='C1', y='C2', data=tmp, s=90, c=my_colors[5], linewidth=0.5,
                edgecolor='k', marker='P', alpha=1)
tmp = d[d.Mutation == 6]
g7 = ax.scatter(x='C1', y='C2', data=tmp, s=90, c=my_colors[6], linewidth=0.5,
                edgecolor='k', marker='D', alpha=1)
tmp = d[d.Mutation == 7]
g8 = ax.scatter(x='C1', y='C2', data=tmp, s=90, c=my_colors[7], linewidth=0.5,
                edgecolor='k', marker='D', alpha=1)
tmp = d[d.Mutation == 0]
g1 = ax.scatter(x='C1', y='C2', data=tmp, s=90, c=my_colors[0], linewidth=0.5,
                edgecolor='k', marker='o', alpha=1)
tmp = d[d.Mutation == 1]
g2 = ax.scatter(x='C1', y='C2', data=tmp, s=90, c=my_colors[1], linewidth=0.5,
                edgecolor='k', marker='o', alpha=1)
ax.legend([(g1, g2), (g3, g4), (g5, g6), (g7, g8)], [category_type[0],
                                                     category_type[1],
                                                     category_type[2],
                                                     category_type[3]],
          scatterpoints=1, ncol=2, fancybox=False, framealpha=1,
          shadow=False, borderpad=0.1, numpoints=1,
          loc='lower right',
          handler_map={tuple: HandlerTuple(ndivide=None)})
ax.set_title('Two-component PCA decomposition', fontsize=23)
ax.set_xlabel('$\mathregular{PC_1}$ explained variance: ' +
              str(np.round(pca.explained_variance_ratio_[0]*100, 1)) +
              '%', fontsize='20')
ax.set_ylabel('$\mathregular{PC_2}$ explained variance: ' +
              str(np.round(pca.explained_variance_ratio_[1]*100, 1))+'%',
              fontsize='20')
ax.set_box_aspect(1)
ax.xaxis.grid()
ax.yaxis.grid()
plt.ylim(bottom=-17)
plt.xlim(left=-30, right=30)
plt.show()


# %% Plot Two-dimensional LDA
d = pd.DataFrame({'C1': X_LDA[:, 0], 'C2': X_LDA[:, 1],
                  'Mutation': y})

sns.set_theme(style="ticks", palette='colorblind', font_scale=1.75)
fig, ax = plt.subplots(1, figsize=(9, 9))
hb = ax.hexbin(model.transform(X_ctrl_ss)[:, 0],
               model.transform(X_ctrl_ss)[:, 1],
               cmap='Greys', gridsize=(9, 6))
cb = fig.colorbar(hb, ax=ax, shrink=0.75,
                  format=tick.FormatStrFormatter('%.0f'))
cb.set_label('Number of controls', size=20)
tmp = d[d.Mutation == 2]
g3 = ax.scatter(x='C1', y='C2', data=tmp, s=90, c=my_colors[2], linewidth=0.5,
                edgecolor='k', marker='^')
tmp = d[d.Mutation == 3]
g4 = ax.scatter(x='C1', y='C2', data=tmp, s=90, c=my_colors[3], linewidth=0.5,
                edgecolor='k', marker='^')
tmp = d[d.Mutation == 4]
g5 = ax.scatter(x='C1', y='C2', data=tmp, s=90, c=my_colors[4], linewidth=0.5,
                edgecolor='k', marker='P')
tmp = d[d.Mutation == 5]
g6 = ax.scatter(x='C1', y='C2', data=tmp, s=90, c=my_colors[5], linewidth=0.5,
                edgecolor='k', marker='P')
tmp = d[d.Mutation == 6]
g7 = ax.scatter(x='C1', y='C2', data=tmp, s=90, c=my_colors[6], linewidth=0.5,
                edgecolor='k', marker='D')
tmp = d[d.Mutation == 7]
g8 = ax.scatter(x='C1', y='C2', data=tmp, s=90, c=my_colors[7], linewidth=0.5,
                edgecolor='k', marker='D')
tmp = d[d.Mutation == 0]
g1 = ax.scatter(x='C1', y='C2', data=tmp, s=90, c=my_colors[0], linewidth=0.5,
                edgecolor='k', marker='o')
tmp = d[d.Mutation == 1]
g2 = ax.scatter(x='C1', y='C2', data=tmp, s=90, c=my_colors[1], linewidth=0.5,
                edgecolor='k', marker='o')
ax.legend([(g1, g2), (g3, g4), (g5, g6), (g7, g8)], [category_type[0],
                                                     category_type[1],
                                                     category_type[2],
                                                     category_type[3]],
          scatterpoints=1, ncol=2, fancybox=False, framealpha=1,
          shadow=False, borderpad=0.1, numpoints=1,
          loc='lower right',
          handler_map={tuple: HandlerTuple(ndivide=None)})
ax.set_title('Two-component LDA decomposition', fontsize=23)
ax.set_xlabel('$\mathregular{LD_1}$', fontsize=20)
ax.set_ylabel('$\mathregular{LD_2}$', fontsize=20)
ax.set_box_aspect(1)
ax.xaxis.grid()
ax.yaxis.grid()
ax.set_yticks([-10, -5, 0, 5, 10])
plt.ylim(bottom=-14)
plt.show()
