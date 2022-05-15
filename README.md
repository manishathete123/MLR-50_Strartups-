# MLR-50_Strartups-
Multi Linear Regression
#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.formula.api as smf
import numpy as np


# In[18]:


#Read the data
data = pd.read_csv("C:\\Users\\Dell\\Downloads\\50_Startups.csv")
data.head()


# In[19]:


data.info()


# In[20]:


#check for missing values
data.isna().sum()


# # Correlation Matrix

# In[21]:


data.corr()


# # Scatterplot between variables along with histograms

# In[22]:


#Format the plot background and scatter plots for all the variables
sns.set_style(style='darkgrid')
sns.pairplot(data)


# # Preparing a model

# In[23]:


data1=data.rename({'R&D Spend':'RDS','Administration':'ADMS','Marketing Spend':'MKTS'},axis=1)
data1


# In[25]:


data1[data1.duplicated()] # No duplicated data


# In[26]:


data1.describe()


# In[27]:


#Build model
import statsmodels.formula.api as smf 
model=smf.ols("Profit~RDS+ADMS+MKTS",data=data1).fit()


# In[28]:


#Model Testing
# Coefficients
model.params


# In[29]:


#t and p-Values
print(model.tvalues, '\n', model.pvalues)


# In[30]:


# Finding rsquared values
model.rsquared , model.rsquared_adj  # Model accuracy is 94.75%


# # Simple Linear Regression Models

# In[31]:


slr_a=smf.ols("Profit~ADMS",data=data1).fit()
slr_a.tvalues , slr_a.pvalues  # ADMS has in-significant pvalue 


# In[32]:


slr_m=smf.ols("Profit~MKTS",data=data1).fit()
slr_m.tvalues , slr_m.pvalues  # MKTS has significant pvalue 


# In[34]:


mlr_am=smf.ols("Profit~ADMS+MKTS",data=data1).fit()
mlr_am.tvalues , mlr_am.pvalues  # varaibles have significant pvalues
 


# # Calculating VIF

# Model Validation
# Two Techniques: 1. Collinearity Check & 2. Residual Analysis

# In[35]:


# 1) Collinearity Problem Check
# Calculate VIF = 1/(1-Rsquare) for all independent variables

rsq_r=smf.ols("RDS~ADMS+MKTS",data=data1).fit().rsquared
vif_r=1/(1-rsq_r)

rsq_a=smf.ols("ADMS~RDS+MKTS",data=data1).fit().rsquared
vif_a=1/(1-rsq_a)

rsq_m=smf.ols("MKTS~RDS+ADMS",data=data1).fit().rsquared
vif_m=1/(1-rsq_m)

# Putting the values in Dataframe format
d1={'Variables':['RDS','ADMS','MKTS'],'Vif':[vif_r,vif_a,vif_m]}
Vif_df=pd.DataFrame(d1)
Vif_df


# # Residual Analysis

# ## Test for Normality of Residuals (Q-Q Plot)

# In[39]:
