#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import preprocessing


# In[3]:


df1 = pd.read_csv('Case Study - Customer.csv', delimiter = ';')
df2 = pd.read_csv('Case Study - Product.csv', delimiter = ';')
df3 = pd.read_csv('Case Study - Store.csv', delimiter = ';')
df4 = pd.read_csv('Case Study - Transaction.csv', delimiter = ';')
print(df4.head())
print(df1.head())


# In[4]:


df = df4.merge(df1, on='CustomerID').merge(df2, on='ProductID').merge(df3, on='StoreID')
df


# In[5]:


df.info()


# In[6]:


# basen on SQL Query
def umur(x):
    if x < 30:
        return 'Single'
    else:
        return 'Married'

df['marital2'] = df['Age'].apply(umur)
#isi nan di kolom marital status
df['Marital Status'] = df['Marital Status'].fillna(df['marital2'])
df.isna().sum()


# In[7]:


df = df.drop(columns=['Latitude', 'Longitude', 'Price_y', 'marital2'], axis = 1)
df.columns


# In[8]:


groupT = df.groupby(['Date'])['TotalAmount'].sum().reset_index()#.sort_values('Date', ascending = False)
groupT


# In[9]:


ax = sns.boxplot(data = df['TotalAmount'])


# In[10]:


# Mengubah kolom Date menjadi tipe data datetime dengan format yang sesuai
groupT['Date'] = pd.to_datetime(groupT['Date'], format='%d/%m/%Y')

# Urutkan DataFrame berdasarkan tanggal
groupT.sort_values(by='Date', inplace=True)


# In[11]:


groupT.head(30)


# In[12]:


#ine chart example
#import matplotlib.ticker as plticker

plt.figure(figsize= (13,5))
plt.style.use('ggplot')
plt.plot(groupT['Date'], groupT['TotalAmount'], linewidth=1.5)
plt.title('Total Amount Graph', fontsize = 20)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(60)) #mengubah interval xticks to 2 years interval
plt.xlabel('Datetime', color = 'b')
plt.ylabel('Total Amount', color = 'b')


# In[13]:


# Menambahkan kolom DayOfYear yang merepresentasikan urutan harian dalam setahun
groupT['DayOfYear'] = groupT['Date'].dt.dayofyear
groupT['Day'] = groupT['Date'].dt.day
groupT


# In[14]:


# Menghasilkan nomor minggu menggunakan dt.week
groupT['weeknum'] = groupT['Date'].dt.week
groupT


# In[15]:


groupT = groupT[~groupT['DayOfYear'].isin([1, 2])]


# In[16]:


groupw = groupT.groupby(['weeknum'])['TotalAmount'].sum().reset_index()
groupw['Year'] = 2022


# In[17]:


ax = sns.boxplot(data = groupw['TotalAmount'])


# no outlier

# In[18]:


groupw['Date'] = pd.to_datetime(groupw['Year'].astype(str) + groupw['weeknum'].astype(str) + '1', format='%G%V%u')
groupw


# In[19]:


groupw = groupw[['weeknum', 'TotalAmount']]


# In[20]:


#stationer or not
from statsmodels.tsa.stattools import adfuller
result = adfuller(groupw['TotalAmount'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
  print('\t%s: %.3f' % (key, value))


# In[26]:


groupw['TotalAmount'] = groupw['TotalAmount'].astype('float')


# In[27]:


train = groupw[['weeknum', 'TotalAmount']].iloc[:-12]
test = groupw[['weeknum', 'TotalAmount']].iloc[-12:]
train.head()

train.columns


# In[28]:


groupw.info()


# In[479]:


plt.figure(figsize= (13,5))
plt.style.use('ggplot')
plt.plot(groupw['weeknum'], groupw['TotalAmount'], linewidth=1.5)
plt.title('Total Amount Graph per Week', fontsize = 20)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(8)) #mengubah interval xticks to 2 years interval
plt.xlabel('Week', color = 'b')
plt.ylabel('Total Amount', color = 'b')


# seasonal = 8, karena umumnya memiliki kenaikan 2x - 3x selama 8 minggu.

# In[32]:


from statsmodels.tsa.seasonal import seasonal_decompose
result_add = seasonal_decompose(x=groupw['TotalAmount'], model='additive', extrapolate_trend='freq', period=12)
plt.rcParams.update({'figure.figsize': (5,5)})
result_add.plot().suptitle('', fontsize=22)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(24))
plt.show()


# seasonal = 4, tapi mari kita buat menjadi 8

# In[43]:


import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

# Original Series
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
ax1.plot(groupw['TotalAmount']); ax1.set_title('Original Series'); ax1.axes.xaxis.set_visible(False)
# 1st Differencing
ax2.plot(groupw['TotalAmount'].diff()); ax2.set_title('1st Order Differencing'); ax2.axes.xaxis.set_visible(False)
# 2nd Differencing
ax3.plot(groupw['TotalAmount'].diff().diff()); ax3.set_title('2nd Order Differencing'); ax3.axes.xaxis.set_visible(False)
# 3rd Differencing
ax4.plot(groupw['TotalAmount'].diff().diff().diff()); ax4.set_title('3rd Order Differencing')
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(4))
plt.show()


# In[51]:


#stationer or not
from statsmodels.tsa.stattools import adfuller
result = adfuller(groupw['TotalAmount'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
  print('\t%s: %.3f' % (key, value))


# data is stationer if p-value < 0.05

# In[100]:


import itertools
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

def acf_pacf(groupw,alags=26,plags=26):
    '''
        Performs acf/pacf results plot

        df          Dataframe to Analyse
    '''

    #Create figure
    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(13,8))
    #Make ACF plot
    plot_acf(groupw['TotalAmount'],lags=alags, zero=False,ax=ax1)
    #Make PACF plot
    plot_pacf(groupw['TotalAmount'],lags=plags, ax=ax2)
    plt.show()

acf_pacf(groupw,alags=25,plags=25)


# p = 0, q = 1, s = 8?

# In[54]:


groupw.columns


# In[56]:


train = groupw[['weeknum', 'TotalAmount']].iloc[:-12]
test = groupw[['weeknum', 'TotalAmount']].iloc[-12:]
test


# In[57]:


train.set_index('weeknum', inplace=True)
test.set_index('weeknum', inplace=True)
test


# In[471]:


from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(train['TotalAmount'], order = (1, 0, 0), seasonal_order=(1, 1, 0, 8))
#this is the best period i've us
sarima = model.fit()
sarima.summary()


# In[190]:


len(train)


# In[195]:


train.tail(2)


# In[191]:


len(test)


# In[194]:


test.head(3)


# In[472]:


start = len(train)+1
end = len(train)+len(test)

predict = sarima.predict(start = start, end = end, dynamic = False, type='level')
predict.reset_index()


# In[473]:


predict.plot(legend = True)
test['TotalAmount'].plot(legend = True)


# In[474]:


from sklearn.metrics import mean_absolute_error as mae
import numpy as np

print(mae(test, predict))

""" Have tried many models, and this is the best model """


# In[475]:


# Hitung selisih antara y_train_scaled dan y_pred_scaled
residuals = predict - test['TotalAmount']

# Plot histogram atau KDE dari selisih (residuals)
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Test and Predicted Difference (thousand)', color = 'darkblue', fontsize = 12)
plt.ylabel('Frequency', color = 'darkblue', fontsize = 12)
xtick_label, location = plt.xticks()
new_xtick_labels = (xtick_label / 1000).astype(int)  # Mengubah skala menjadi jutaan dan integer
plt.xticks(xtick_label, new_xtick_labels, rotation=0)
plt.title('Distribution of Test and Predicted Difference', pad = 10, color = 'darkred', fontsize = 17)
#plt.savefig('Distribution_of_Residuals_by_Linear_Regression_Model_in_East_Jakarta.jpg', bbox_inches='tight')
plt.show()


# # Predict next 7 week

# In[511]:


model2 = SARIMAX(groupw['TotalAmount'], order = (1, 0, 0), seasonal_order=(1, 1, 0, 8))
#this is the best period i've us
sarima2 = model2.fit()
sarima2.summary()


# In[528]:


start = len(train)+len(test)+1
end = start + 11

predict2 = sarima2.predict(start = start, end = end, dynamic = False, type='level')
predict2 = predict2.reset_index()


# In[524]:


predict2.columns


# In[529]:


plt.figure(figsize= (13,5))
plt.style.use('ggplot')
plt.plot(predict2['index'], predict2['predicted_mean'], linewidth=1.5)
plt.title('Predicted Total Amount Graph in first 3 months of 2023', fontsize = 20, pad = 10)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(2)) #mengubah interval xticks to 2 years interval
plt.xlabel('Week', color = 'b')
plt.ylabel('Total Amount', color = 'b')

