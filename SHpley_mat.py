#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler , StandardScaler , MaxAbsScaler , Normalizer
from sklearn.model_selection import train_test_split
from pandas.core.common import random_state
from sklearn.metrics import r2_score
import warnings
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
import math
warnings.filterwarnings("ignore")


# In[ ]:


df = pd.read_excel(r"D:\Articles\Mat anchorage article\2.xlsx",sheet_name='Failure' ,header = 0 )
y = df.loc[:, 'Failure mode'].to_numpy().reshape((-1, 1))
X = df.iloc[:, [0, 1, 2, 3, 4]].to_numpy()
Xtr,Xte,ytr,yte = train_test_split(X,y, test_size=0.3, random_state=42)
model = XGBClassifier()
model.fit(Xtr , ytr)


# In[ ]:


features=['f\u02b9c (MPa)','Type of anchor',' Length of anchor (mm)',' Anchor diameter (mm)', 'Number of anchors']
c=['purple','orange','b','red','green']
explainer = shap.Explainer(model,X,feature_names=features)
shap_values = explainer(X)


# In[ ]:



import shap

# Generate Shapley values
shap_values = explainer(X)

# Create a figure and axis object using matplotlib
fig, ax = plt.subplots(figsize=(12, 8))

# Create a beeswarm plot using shap.plots.beeswarm()
shap.plots.beeswarm(shap_values, show=False)

# Customize the plot using matplotlib functions
ax.set_xlabel('SHapley values:impact on model output  \n\n (b)', fontsize=12, fontname='Times New Roman',fontweight='bold')
ax.set_ylabel('Features', fontsize=14, fontname='Times New Roman',fontweight='bold')
# ax.set_title('Shapley Values', fontsize=16, fontname='Times New Roman')

# Set the font of the legend
handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(handles, labels, loc='upper right', fontsize=12)
for text in legend.texts:
    text.set_fontname('Times New Roman')

# Set the font of the ticks
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontname('Times New Roman')
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontname('Times New Roman')

# Save the figure
fig.savefig('shapley_values.png', dpi=400, bbox_inches='tight')


# In[ ]:


df = pd.read_excel(r"D:\Articles\Mat anchorage article\2.xlsx",sheet_name='Failure' ,header = 0 )
y = df.loc[:, 'Fs (MPa)'].to_numpy().reshape((-1, 1))
X = df.iloc[:, [0, 1, 2, 3, 4]].to_numpy()
Xtr,Xte,ytr,yte = train_test_split(X,y, test_size=0.3, random_state=42)
model = XGBRegressor()
model.fit(Xtr , ytr)
features=['f\u02b9c (MPa)','Type of anchor',' Length of anchor (mm)',' Anchor diameter (mm)', 'Number of anchors']
c=['purple','orange','b','red','green']
explainer = shap.Explainer(model,X,feature_names=features)
shap_values = explainer(X)


# In[ ]:


# Generate Shapley values
shap_values = explainer(X)

# Create a figure and axis object using matplotlib
fig, ax = plt.subplots(figsize=(12, 8))

# Create a beeswarm plot using shap.plots.beeswarm()
shap.plots.beeswarm(shap_values, show=False)

# Customize the plot using matplotlib functions
ax.set_xlabel('SHapley values:impact on model output  \n\n (a)', fontsize=12, fontname='Times New Roman',fontweight='bold')
ax.set_ylabel('Features', fontsize=14, fontname='Times New Roman',fontweight='bold')
# ax.set_title('Shapley Values', fontsize=16, fontname='Times New Roman')

# Set the font of the legend
handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(handles, labels, loc='upper right', fontsize=12)
for text in legend.texts:
    text.set_fontname('Times New Roman')

# Set the font of the ticks
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontname('Times New Roman')
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontname('Times New Roman')

# Save the figure
fig.savefig('shapley_values.png', dpi=400, bbox_inches='tight')


# In[ ]:


# Generate Shapley values
shap_values = explainer(Xtr)

# Create a figure and axis object using matplotlib
fig, ax = plt.subplots(figsize=(20, 15))

# Create a beeswarm plot using shap.plots.beeswarm()
shap.summary_plot(shap_values, Xte, plot_type="bar",axis_color='black',color=c,feature_names=features , show=False)

# Customize the plot using matplotlib functions
ax.set_xlabel('SHapley values:impact on model output  \n\n ', fontsize=12, fontname='Times New Roman',fontweight='bold')
ax.set_ylabel('Features', fontsize=14, fontname='Times New Roman',fontweight='bold')
# ax.set_title('Shapley Values', fontsize=16, fontname='Times New Roman')

# Set the font of the legend
handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(handles, labels, loc='upper right', fontsize=12)
for text in legend.texts:
    text.set_fontname('Times New Roman')

# Set the font of the ticks
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontname('Times New Roman')
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontname('Times New Roman')

# Save the figure
fig.savefig('shapley_values.png', dpi=300, bbox_inches='tight')

