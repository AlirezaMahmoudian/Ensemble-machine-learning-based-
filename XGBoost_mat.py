#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler , StandardScaler , MaxAbsScaler , Normalizer
from sklearn.model_selection import train_test_split
from pandas.core.common import random_state
import sklearn.decomposition as dec
from sklearn.linear_model import SGDRegressor , Ridge , LinearRegression , Lasso , LassoLars
from sklearn.metrics import r2_score
import warnings
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor , RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")
from random import shuffle
from random import randint as rnd
import random


# In[37]:


df = pd.read_excel(r"D:\Articles\Mat anchorage article\2.xlsx"  ,header = 0 )
y = df.loc[:, 'Fs (MPa)'].to_numpy().reshape((-1, 1))
X = df.iloc[:, [0, 1, 2, 3, 4]].to_numpy()
Xtr,Xte,ytr,yte = train_test_split(X,y, test_size=0.3, random_state=42)


# In[15]:


model=XGBRegressor(random_state=42)
model.fit(Xtr , ytr)
yprtr = model.predict(Xtr)
yprte = model.predict(Xte)
r2tr=round(r2_score(ytr , yprtr),2)
r2te=round(r2_score(yte , yprte),2)
print(r2tr,r2te)
msetr=round(mean_squared_error(ytr , yprtr)**0.5,3)
msete=round(mean_squared_error(yte , yprte)**0.5,3)

# msetr=round(mean_squared_error(ytr , yprtr),3)
# msete=round(mean_squared_error(yte , yprte),3)

a = min([np.min(ytr), np.min(yte), 0])
b = max([np.max(ytr), np.max(yte), 1])

# plt.subplot(1, 2, 1)
plt.scatter(ytr, yprtr, s=80, facecolors='mediumseagreen', edgecolors='black')
plt.plot([a, b], [a, b], c='gray', lw=1.4, label='y = x')
plt.title(f'XGboost Train [R2 = {round(r2tr, 3)}] & [RMSE = {round(msetr, 2)}] ',fontsize=14)
plt.xlabel('Fs (MPa)_real',fontsize=14)
plt.ylabel('Fs (MPa)_predicted',fontsize=14)
plt.legend()
plt.show()

# plt.subplot(1, 2, 2)
plt.scatter(yte, yprte, s=80, facecolors='mediumseagreen', edgecolors='black')
plt.plot([a, b], [a, b], c='gray', lw=1.4, label='y = x')
plt.title(f'XGboost Test [R2 = {round(r2te, 2)}] & [RMSE = {round(msete, 2)}] ',fontsize=14)
plt.xlabel('Fs (MPa)_real',fontsize=14)
plt.ylabel('Fs (MPa)_predicted',fontsize=14)
plt.legend()
plt.show()


# In[39]:


df = pd.read_excel(r"D:\Articles\Mat anchorage article\2.xlsx"  ,header = 0 )
y = df.loc[:, 'Fs (MPa)'].to_numpy().reshape((-1, 1))
X = df.iloc[:, [0, 1, 2, 3, 4]].to_numpy()
Xtr,Xte,ytr,yte = train_test_split(X,y, test_size=0.3, random_state=42)


# In[40]:


n = 4                          #Number of Queens
p = 500                        #Number of Population
m=10                           #Number of Steps
mr=0.7
epoch=200


# In[41]:


def randomGeneration(NumberOfRows,NumberOfQueens,m):
    generation_list = []
    for i in range(NumberOfRows):
        gene = []
        for j in range(NumberOfQueens):
            gene.append(random.randint(1,m))
        generation_list.append(gene)
    return generation_list
def cross_over(generation_list,p,n):
    for i in range(0,p,2):
        child1=generation_list[i][:n//2]+generation_list[i+1][n//2:n]
        child2=generation_list[i+1][:n//2]+generation_list[i][n//2:n]
        generation_list.append(child1)
        generation_list.append(child2)
    return generation_list
def mutation(generation_list,p,n,m,mr):
    chosen_ones = list(range(p,p*2))
    shuffle(chosen_ones)
    chosen_ones = chosen_ones[:int(p*mr)]
    
    for i in chosen_ones:
        cell = rnd(0,n-1)
        val = rnd (1,m)
        generation_list[i][cell] = val
    return generation_list
def fitness(population_list):
    fitness=[]
    for i in population_list:
        member=[]
        model = XGBRegressor( n_estimators = i[0]*30 , learning_rate = i[1]/10 ,
                              max_depth = i[2]+1 , subsample = i[3]/10 )
                                                                                
        model.fit(Xtr , ytr)
        yprte = model.predict(Xte)
        r2te=round(r2_score(yte , yprte),2)
        member.extend(i)
        member.append(r2te)
        fitness.append(member)
    return fitness
def hazf(result):
    for i in result:
        i.pop()
    return result


# In[42]:


pop=randomGeneration(p,n,m)
for i in range (epoch):
    pop = cross_over(pop,p,n)
    pop = mutation(pop,p,n,m,mr)
    pop = fitness(pop)
    pop.sort(key=lambda x: x[-1],reverse=True)
    pop = pop[:p]
    print('The best solution so far:' , pop[0])
    print('The worse solution so far:' , pop[-1])
    pop=hazf(pop)


# In[11]:


model = XGBRegressor( n_estimators = 30 , learning_rate = 0.5 ,
                     max_depth = 3 , subsample = 0.3 , random_state=0  )
model.fit(Xtr , ytr)
yprtr = model.predict(Xtr)
yprte = model.predict(Xte)
r2tr=round(r2_score(ytr , yprtr),2)
r2te=round(r2_score(yte , yprte),2)
print(r2tr,r2te)
msetr=round(mean_squared_error(ytr , yprtr)**0.5,3)
msete=round(mean_squared_error(yte , yprte)**0.5,3)

# msetr=round(mean_squared_error(ytr , yprtr),3)
# msete=round(mean_squared_error(yte , yprte),3)

a = min([np.min(ytr), np.min(yte), 0])
b = max([np.max(ytr), np.max(yte), 1])

# plt.subplot(1, 2, 1)
plt.scatter(ytr, yprtr, s=80, facecolors='mediumseagreen', edgecolors='black')
plt.plot([a, b], [a, b], c='gray', lw=1.4, label='y = x')
plt.title(f'XGBoost tree Train [R2 = {round(r2tr, 3)}] & [RMSE = {round(msetr, 2)}] ',fontsize=14)
plt.xlabel('Fs (MPa)_real',fontsize=14)
plt.ylabel('Fs (MPa)_predicted',fontsize=14)
plt.legend()
plt.show()

# plt.subplot(1, 2, 2)
plt.scatter(yte, yprte, s=80, facecolors='mediumseagreen', edgecolors='black')
plt.plot([a, b], [a, b], c='gray', lw=1.4, label='y = x')
plt.title(f'XGBoost Test [R2 = {round(r2te, 2)}] & [RMSE = {round(msete, 2)}] ',fontsize=14)
plt.xlabel('Fs (MPa)_real',fontsize=14)
plt.ylabel('Fs (MPa)_predicted',fontsize=14)
plt.legend()
plt.show()


# In[43]:


a=9
b=5
c=49
d=9


# In[44]:


df = pd.read_excel(r"D:\Articles\Mat anchorage article\2.xlsx"  ,header = 0 )
y = df.loc[:, 'Fs (MPa)'].to_numpy().reshape((-1, 1))
X = df.iloc[:, [0, 1, 2, 3, 4]].to_numpy()
Xtr,Xte,ytr,yte = train_test_split(X,y, test_size=0.3, random_state=42)
model = XGBRegressor( n_estimators = a*30 , learning_rate = b/10 ,
                              max_depth = c+1 , subsample = d/10 ) 
model.fit(Xtr , ytr)
yprtr = model.predict(Xtr)
yprte = model.predict(Xte)
r2tr=round(r2_score(ytr , yprtr),2)
r2te=round(r2_score(yte , yprte),2)
print(r2tr,r2te)
msetr=round(mean_squared_error(ytr , yprtr)**0.5,3)
msete=round(mean_squared_error(yte , yprte)**0.5,3)

# msetr=round(mean_squared_error(ytr , yprtr),3)
# msete=round(mean_squared_error(yte , yprte),3)

a = min([np.min(ytr), np.min(yte), 0])
b = max([np.max(ytr), np.max(yte), 1])

# plt.subplot(1, 2, 1)
plt.scatter(ytr, yprtr, s=80, facecolors='mediumseagreen', edgecolors='black')
plt.plot([a, b], [a, b], c='gray', lw=1.4, label='y = x')
plt.title(f'XGBoost tree Train [R2 = {round(r2tr, 3)}] & [RMSE = {round(msetr, 2)}] ',fontsize=14)
plt.xlabel('Fs (MPa)_real',fontsize=14)
plt.ylabel('Fs (MPa)_predicted',fontsize=14)
plt.legend()
plt.show()

# plt.subplot(1, 2, 2)
plt.scatter(yte, yprte, s=80, facecolors='mediumseagreen', edgecolors='black')
plt.plot([a, b], [a, b], c='gray', lw=1.4, label='y = x')
plt.title(f'XGBoost Test [R2 = {round(r2te, 2)}] & [RMSE = {round(msete, 2)}] ',fontsize=14)
plt.xlabel('Fs (MPa)_real',fontsize=14)
plt.ylabel('Fs (MPa)_predicted',fontsize=14)
plt.legend()
plt.show()


# In[ ]:




