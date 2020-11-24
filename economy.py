#!/usr/bin/env python
# coding: utf-8

# ## <center>DOES FOREIGN AID PROMOTE ECONOMIC GROWTH?<center>
#     

# In[1]:


##importing all the dictionries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from scipy import stats
import statsmodels.api as sm

import statsmodels.formula.api as smf
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm


# In[114]:


data = pd.read_excel("workingData_big.xlsx", sheet_name="Low income",  na_values=["."])


# In[115]:


data.head()


# In[4]:


data.shape


# In[5]:


data.columns


# <H2><center>DATA CLEANING</center></H2>

# In[6]:


#removing columns that are not required
data_1=data.drop(columns=['Country Code','Time Code','GDP growth (annual %)',
                          'Domestic credit to private sector (% of GDP) [FS.AST.PRVT.GD.ZS]'
                         ,'Consumer price index (2010 = 100) [FP.CPI.TOTL]',
                          'ln_initial2000_gdp','GDP per capita growth (annual %)','ln_gdp'
                         ,'Gross savings (% of GDP)'], axis=1)
##renaming columns 
data_new = data_1.rename(columns = {'Country Name':'Country','Trade (% of GDP) [NE.TRD.GNFS.ZS]':'Trade',
             'Population, total [SP.POP.TOTL]':'Total_Population',
             'Imports of goods and services (% of GDP) [NE.IMP.GNFS.ZS]':'Imports',
                                   'Exports of goods and services (% of GDP) [NE.EXP.GNFS.ZS]':'Exports',
                                   'Trade (% of GDP) [NE.TRD.GNFS.ZS]':'Trade'
                                   ,'Net official development assistance (constant 2015 US$)':'ODA'
                        ,'Gross fixed capital formation (% of GDP) [NE.GDI.FTOT.ZS]':'Gross fixed capital formation (% of GDP)'
                                   ,'Gross domestic savings (% of GDP) [NY.GDS.TOTL.ZS]':'Gross domestic savings (% of GDP)'
                                   })


# In[7]:


# replace missing values with means of their respective countries
data_new['Trade'] = data_new['Trade'].fillna(data_new.groupby('Country')['Trade'].transform('mean'))
data_new['Trade'] = data_new['Trade'].fillna(data_new['Trade'].mean())

data_new['GDP (constant 2010 US$)'] = data_new['GDP (constant 2010 US$)'].fillna(data_new.groupby('Country')['GDP (constant 2010 US$)'].transform('mean'))
data_new['GDP (constant 2010 US$)'] = data_new['GDP (constant 2010 US$)'].fillna(data_new['GDP (constant 2010 US$)'].mean())

data_new['GDP per capita (constant 2010 US$)'] = data_new['GDP per capita (constant 2010 US$)'].fillna(data_new.groupby('Country')['GDP per capita (constant 2010 US$)'].transform('mean'))
data_new['GDP per capita (constant 2010 US$)'] = data_new['GDP per capita (constant 2010 US$)'].fillna(data_new['GDP per capita (constant 2010 US$)'].mean())

data_new['Exports'] = data_new['Exports'].fillna(data_new.groupby('Country')['Exports'].transform('mean'))
data_new['Exports'] = data_new['Exports'].fillna(data_new['Exports'].mean())

data_new['Imports'] = data_new['Imports'].fillna(data_new.groupby('Country')['Imports'].transform('mean'))
data_new['Imports'] = data_new['Imports'].fillna(data_new['Imports'].mean())

data_new['Aid/Gdp'] = data_new['Aid/Gdp'].fillna(data_new.groupby('Country')['Aid/Gdp'].transform('mean'))
data_new['Aid/Gdp'] = data_new['Aid/Gdp'].fillna(data_new['Aid/Gdp'].mean())

data_new['Aid/Gdp_sqr'] = data_new['Aid/Gdp_sqr'].fillna(data_new.groupby('Country')['Aid/Gdp_sqr'].transform('mean'))
data_new['Aid/Gdp_sqr'] = data_new['Aid/Gdp_sqr'].fillna(data_new['Aid/Gdp_sqr'].mean())

data_new['ln_ODA'] = data_new['ln_ODA'].fillna(data_new.groupby('Country')['ln_ODA'].transform('mean'))
data_new['ln_ODA'] = data_new['ln_ODA'].fillna(data_new['ln_ODA'].mean())

data_new['ODA'] = data_new['ODA'].fillna(data_new.groupby('Country')['ODA'].transform('mean'))
data_new['ODA'] = data_new['ODA'].fillna(data_new['ODA'].mean())

data_new['wopen'] = data_new['wopen'].fillna(data_new.groupby('Country')['wopen'].transform('mean'))
data_new['wopen'] = data_new['wopen'].fillna(data_new['wopen'].mean())

data_new['Gross fixed capital formation (% of GDP)'] = data_new['Gross fixed capital formation (% of GDP)'].fillna(data_new.groupby('Country')['Gross fixed capital formation (% of GDP)'].transform('mean'))
data_new['Gross fixed capital formation (% of GDP)'] = data_new['Gross fixed capital formation (% of GDP)'].fillna(data_new['Gross fixed capital formation (% of GDP)'].mean())

data_new['Gross domestic savings (% of GDP)'] = data_new['Gross domestic savings (% of GDP)'].fillna(data_new.groupby('Country')['Gross domestic savings (% of GDP)'].transform('mean'))
data_new['Gross domestic savings (% of GDP)'] = data_new['Gross domestic savings (% of GDP)'].fillna(data_new['Gross domestic savings (% of GDP)'].mean())

data_new['FDI, net inflows (% of GDP)'] = data_new['FDI, net inflows (% of GDP)'].fillna(data_new.groupby('Country')['ODA'].transform('mean'))
data_new['FDI, net inflows (% of GDP)'] = data_new['FDI, net inflows (% of GDP)'].fillna(data_new['FDI, net inflows (% of GDP)'].mean())


# In[8]:


#descriptive stats
data_new.describe()


# In[9]:


data_new.isnull().sum()


# <H2><CENTER>DESCRIPTIVE ANALYSIS<CENTER></H2>

# In[52]:


## AID over the years 1990-2017(Bar Chart)
sns.set_style('whitegrid')
ODA_1990_2017 = sns.barplot(x='Time', y='ODA', data=data_new, errwidth=0, capsize=10)
plt.xticks(rotation=70)
plt.rcParams["xtick.labelsize"]


# In[53]:


## AID over the years 1990-2017(Bar Chart)
sns.set_style('whitegrid')
ODA_1990_2017 = sns.barplot(x='Time', y='GDP per capita (constant 2010 US$)', data=data_new, errwidth=0, capsize=10)
plt.xticks(rotation=70)
plt.rcParams["xtick.labelsize"]


# In[12]:


## GDP per capita vs ODA(Joint Map)
sns.set(style="white")
# Generate a random correlated bivariate dataset
rs = np.random.RandomState(5)
mean = [0, 0]
cov = [(1, .5), (.5, 1)]
GDPpc, ODA = rs.multivariate_normal(mean, cov, 500).T
GDPpc = pd.Series(GDPpc, name="$GDP per capita (constant 2010 US$)$")
ODA = pd.Series(ODA, name="$ODA$")

# Show the joint distribution using kernel density estimation
g = sns.jointplot(GDPpc, ODA, kind="kde", height=7, space=0)


# In[13]:


##Correlation matrix heat map
corr = data_new.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(200, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=70,
    horizontalalignment='right'
);


# In[14]:


#correlation matrix
rs = np.random.RandomState(0)
df = pd.DataFrame(rs.rand(10, 10))
corr = data_new.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[15]:


##Making bins for diffrent time periods
group_1 = data_new['Time'].between(1990,2000,inclusive=True)
group_2 = data_new['Time'].between(2000,2010,inclusive=True)
group_3 = data_new['Time'].between(2010,2017,inclusive=True)
group_4 = data_new['Time'].between(2000,2017,inclusive=True)
group_5 = data_new['Time'].between(1990,2017,inclusive=True)


# In[16]:


pre_ODA = sns.relplot(x='Time', y='ODA', hue="Country", data=data_new[group_1], height=5, aspect=1.5)


# In[17]:


post_ODA = sns.relplot(x='Time', y='ODA', hue="Country", data=data_new[group_4], height=5, aspect=1.5)
post_ODA = ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)


# <h1><center>Linear Regression</center></h1>

# In[116]:


data_new[group_5].head()


# In[117]:


# Creating a dataset for regression analysis
entity = data_new[group_5]['Country'].unique()
time = list(pd.date_range('1-1-1990',freq='A', periods=28))
index = pd.MultiIndex.from_product([entity, time])
allvars = ['Gross fixed capital formation (% of GDP)','Gross domestic savings (% of GDP)',
           'Population growth (annual %)','FDI, net inflows (% of GDP)',
            'Aid/Gdp_sqr','Aid/Gdp', 'ln_ODA',
       'ln_gdp_pc', 'wopen','Trade']

df = pd.DataFrame(np.array(data_new[group_5][allvars]),
                  index=index, columns = allvars)


# In[118]:


from linearmodels.panel import PanelOLS
# fixed effects
# documentation: https://bashtage.github.io/linearmodels/panel/models.html#linearmodels.panel.model.PanelOLS

independent_vars = ['Gross fixed capital formation (% of GDP)','Gross domestic savings (% of GDP)',
          'Population growth (annual %)','FDI, net inflows (% of GDP)',
            'Aid/Gdp_sqr','Aid/Gdp', 'ln_ODA','wopen','Trade']

mod = PanelOLS(df['ln_gdp_pc'], 
               df[independent_vars], 
               entity_effects=True, time_effects=True) # you can turn on or off both entity_effects and time_effects

res = mod.fit(cov_type='clustered', cluster_entity=True) # here cov_type means covariance estimators type.
# cov_type can be ‘unadjusted’, ‘homoskedastic’ or ‘robust’, ‘heteroskedastic’ or ‘clustered` - One or two way clustering.

print(res)


# <h4> ODA - Official development assistance
#  Three significant Independent Varibles: Aid/GDP, Aid/GDP^2, ln_ODA </h4>

# <h2><center>DIAGNOSTIC ANALYSIS</center></h2>

# <h3> 1. Unit root test </h3>
# 
# The Augmented Dickey-Fuller test is a type of statistical test called a unit root test
# The null hypothesis of the test is that the time series can be represented by a unit root, that it is not stationary (has some time-dependent structure). The alternate hypothesis (rejecting the null hypothesis) is that the time series is stationary.
# 
# H0: If failed to be rejected, it suggests the time series has a unit root, meaning it is non-stationary. It has some time dependent structure.
#     
# H1: The null hypothesis is rejected; it suggests the time series does not have a unit root, meaning it is stationary. It does not have time-dependent structure.

# In[46]:


##Unit root test on our dependent variable
X=df['ln_gdp_pc'].values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# <h4>For our dependent variable, p-value is less than 0.05. We reject the null hypothesis and say there is no unit root and data is stationary</h4>

# In[47]:


##Unit root test on our independent varible
for col in independent_vars:
    X=df[col].dropna().values
    result = adfuller(X)
    print()
    print('column name:',col, '\nADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])


# <h4>All of the ADF for independent variables have p-value of less than 0.05, which means they are stationary.</h4>

# <h3>2. VIF for multicollinearity</h3>
# 
# Colinearity is the state where two variables are highly correlated and contain similiar information about the variance within a given dataset. To detect colinearity among variables, simply create a correlation matrix and find variables with large absolute values.
# 
# A VIF between 5 and 10 indicates high correlation that may be problematic.

# In[48]:


from statsmodels.tools.tools import add_constant
X = add_constant(df[independent_vars].dropna())
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif


# <h3>3. Homoscedasticity</h3>
# 
# When residuals do not have constant variance (they exhibit heteroscedasticity), it is difficult to determine the true standard deviation of the forecast errors, usually resulting in confidence intervals that are too wide/narrow. For example, if the variance of the residuals is increasing over time, confidence intervals for out-of-sample predictions will be unrealistically narrow.

# In[49]:


from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import het_white

# breuschpagan test
bp_test = het_breuschpagan(res.resids, df[independent_vars].dropna())
labels = ['BP Statistic', 'BP-Test p-value', 'F-Statistic', 'F-Test p-value']
print(pd.Series(zip(labels, bp_test)))


# In[50]:


# visualization of heteroscedasticity
fitted = res.fitted_values
residuals = res.resids
plt.figure(figsize=(12,8))
sns.regplot(x=fitted,  y = residuals)
plt.xlabel('Fitted Values')
plt.xlim([4,6])
plt.show()


# <center><h2>WEIGHTED REGRESSION</h2></center>

# In[51]:


##Weighted regression minimizes the sum of the weighted squared residuals. 
##When you use the correct weights, heteroscedasticity is replaced by homoscedasticity.

from linearmodels.panel import PanelOLS
# fixed effects
# documentation: https://bashtage.github.io/linearmodels/panel/models.html#linearmodels.panel.model.PanelOLS

independent_vars = ['Gross fixed capital formation (% of GDP)','Gross domestic savings (% of GDP)',
          'Population growth (annual %)','FDI, net inflows (% of GDP)',
          'Aid/Gdp', 'Aid/Gdp_sqr', 'ln_ODA','wopen','Trade']

mod = PanelOLS(df['ln_gdp_pc'], 
               df[independent_vars], 
               entity_effects=True, time_effects=True) # you can turn on or off both entity_effects and time_effects

res = mod.fit(cov_type='heteroskedastic') # here cov_type means covariance estimators type.
# cov_type can be ‘unadjusted’, ‘homoskedastic’ or ‘robust’, ‘heteroskedastic’ or ‘clustered` - One or two way clustering.

print(res)


# <h4>Our model was improved, considering the hetroscadacity problems we recieved 5 significant varibles Gross domestic savings(% of GDP), Aid/Gdp, Aid/Gdp_sq, ln_ODA, Trade. </h4>

# In[ ]:




