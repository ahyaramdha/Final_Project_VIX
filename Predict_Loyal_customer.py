#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import preprocessing


# In[2]:


df1 = pd.read_csv('Case Study - Customer.csv', delimiter = ';')
df2 = pd.read_csv('Case Study - Product.csv', delimiter = ';')
df3 = pd.read_csv('Case Study - Store.csv', delimiter = ';')
df4 = pd.read_csv('Case Study - Transaction.csv', delimiter = ';')
print(df4.head())
print(df1.head())


# In[3]:


df = df4.merge(df1, on='CustomerID').merge(df2, on='ProductID').merge(df3, on='StoreID')
df


# In[4]:


df.info()


# In[5]:


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


# In[6]:


df = df.drop(columns=['Latitude', 'Longitude', 'Price_y', 'marital2'], axis = 1)
df.columns


# In[7]:


custcount = df['CustomerID'].value_counts().reset_index()
custcount


# ingin memisahkan kustomer yang loyal (datang tiap bulannya/lebih dari sama dengan 12) dan tidak loyal

# In[8]:


# 1 loyal, 0 tidak loyal
def loyal(x):
    if x >= 12:
        return 1
    else :
        return 0

custcount['loyal'] = custcount['CustomerID'].apply(loyal)
custcount


# In[9]:


custcount = custcount.rename(columns = {'CustomerID': 'kunjungan', 'index':'CustomerID'})
custcount


# In[10]:


groupT = df.groupby(['CustomerID'])['TotalAmount'].median().reset_index()
groupT


# In[11]:


grouprod = df.groupby(['CustomerID'])['Product Name'].value_counts().unstack().reset_index().fillna(0)
grouprod


# In[12]:


cust = custcount.merge(df1, on='CustomerID').merge(grouprod, on = 'CustomerID').merge(groupT, on = 'CustomerID')
cust


# In[13]:


cust = cust.rename(columns = {'TotalAmount':'avg amount'})


# In[14]:


cust.columns


# In[15]:


cust['buytot'] = cust['Cashew'] + cust['Cheese Stick'] + cust['Coffee Candy'] + cust['Crackers'] + cust['Ginger Candy '] + cust['Oat'] + cust['Potato Chip'] + cust['Thai Tea'] + cust['Yoghurt']
cust


# disini kita ingin melihat bagaimana perilaku/ciri kustomer yang loyal dan tidak loyal

# In[16]:


status_onehot = pd.get_dummies(cust["Marital Status"], prefix= '_')
cust = cust.join(status_onehot)
cust.head(2)


# In[17]:


cust['Income'] = cust['Income'].str.replace(',', '.')
cust['Income'] = cust['Income'].astype('float')
cust['Income'] = cust['Income']*1000000
cust.head(2)


# # Handle Outlier

# In[18]:


ax = sns.boxplot(data = cust['avg amount'])


# In[19]:


ax3 = sns.boxplot(data = cust['Income'])


# In[20]:


ax4 = sns.boxplot(data = cust['buytot'])


# In[21]:


#making function to replace outlier
def outlier (x):
    sorted(x)
    q1, q3 = x.quantile([0.25, 0.75])
    IQR = q3 - q1
    lwr_bound = q1 - (1.5*IQR)
    upr_bound = q3 + (1.5*IQR)
    return lwr_bound, upr_bound


# In[22]:


low, high = outlier(cust['Income'])
low2, high2 = outlier(cust['buytot'])
low3, high3 = outlier(cust['avg amount'])


# In[23]:


#replacing outlier with upper bound and lower bound value
cust['Income'] = np.where(cust['Income']>high, high, cust['Income'])
cust['Income'] = np.where(cust['Income']<low, low, cust['Income'])
cust['buytot'] = np.where(cust['buytot']>high2, high2, cust['buytot'])
cust['buytot'] = np.where(cust['buytot']<low2, low2, cust['buytot'])
cust['avg amount'] = np.where(cust['avg amount']>high3, high3, cust['avg amount'])
cust['avg amount'] = np.where(cust['avg amount']<low3, low3, cust['avg amount'])


# In[24]:


ax3 = sns.boxplot(data = cust['Income'])


# # Regresi

# In[25]:


cust['Income'] = cust['Income'].astype('float')
cust.dtypes


# In[26]:


x = cust.drop(columns ={'CustomerID', 'kunjungan', 'loyal', 'Marital Status'}, axis = 1)
y = cust['loyal']
x.head(3)


# In[27]:


from sklearn.model_selection import train_test_split

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


# In[28]:


#persiapan
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_validate

def eval_classification(model):
    y_pred_test = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    y_pred_proba_test = model.predict_proba(x_test)
    y_pred_proba_train = model.predict_proba(x_train)

    print("Accuracy (Train Set): %.2f" % accuracy_score(y_train, y_pred_train))
    print("Precision (Train Set): %.2f" % precision_score(y_train, y_pred_train))
    print("Recall (Train Set): %.2f" % recall_score(y_train, y_pred_train))
    print("F1-Score (Train Set): %.2f" % f1_score(y_train, y_pred_train))

    print("Accuracy (Test Set): %.2f" % accuracy_score(y_test, y_pred_test))
    print("Precision (Test Set): %.2f" % precision_score(y_test, y_pred_test))
    print("Recall (Test Set): %.2f" % recall_score(y_test, y_pred_test))
    print("F1-Score (Test Set): %.2f" % f1_score(y_test, y_pred_test))

    print("roc_auc (test-proba): %.2f" % roc_auc_score(y_test, y_pred_proba_test[:, 1]))
    print("roc_auc (train-proba): %.2f" % roc_auc_score(y_train, y_pred_proba_train[:, 1]))

    score = cross_validate(model, x_train, y_train, cv=5, scoring='roc_auc', return_train_score=True)
    print('roc_auc (crossval train): '+ str(score['train_score'].mean()))
    print('roc_auc (crossval test): '+ str(score['test_score'].mean()))


# ## Random Forest Classifier

# In[29]:


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(x_train, y_train)
y_pred = rf_model.predict(x_test)


# In[30]:


#validasi
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse:.2f}')


# In[31]:


eval_classification(rf_model)


# In[32]:


# Buat confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Tampilkan confusion matrix dalam bentuk heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No', 'loyal'],
            yticklabels=['No', 'loyal'])
plt.xlabel('Predicted value')
plt.ylabel('Real Value')
plt.title('Confusion Matrix of Random Forest Model', pad = 10)
plt.show()


# In[33]:


# Dapatkan fitur-fitur penting dari model
feature_importances = rf_model.feature_importances_

# Urutkan fitur-fitur penting dalam urutan menurun
sorted_indices = feature_importances.argsort()[::-1]

top_n = 10

# Tampilkan fitur-fitur penting
top_features = [x.columns[idx] for idx in sorted_indices[:top_n]]
other_features = [x.columns[idx] for idx in sorted_indices[top_n:]]

# Menggabungkan "Others" untuk fitur yang tidak termasuk dalam top 10
feature_labels = top_features + ['Others']
feature_importances = [feature_importances[idx] for idx in sorted_indices[:top_n]] + [sum(feature_importances[idx] for idx in sorted_indices[top_n:])]

# Visualisasikan fitur-fitur penting
plt.figure(figsize=(10, 6))
plt.barh(range(len(feature_labels)), feature_importances)
plt.yticks(range(len(feature_labels)), feature_labels)
plt.xlabel('Nilai Penting')
plt.ylabel('Fitur')
plt.title('Fitur Penting dari Random Forest',  pad = 15, fontsize = 20, color = 'red')
plt.tight_layout()
plt.show()


# ## Decision Tree Classifier

# In[34]:


from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state = 42)
dt_model.fit(x_train, y_train)
y_pred = dt_model.predict(x_test)


# In[35]:


#validasi
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse:.2f}')


# In[36]:


eval_classification(dt_model)


# In[39]:


# Buat confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Tampilkan confusion matrix dalam bentuk heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No', 'loyal'],
            yticklabels=['No', 'loyal'])
plt.xlabel('Predicted value')
plt.ylabel('Real value')
plt.title('Confusion Matrix of Decision Tree Model', pad = 10)
plt.show()


# In[41]:


# Dapatkan fitur-fitur penting dari model
feature_importances = dt_model.feature_importances_

# Urutkan fitur-fitur penting dalam urutan menurun
sorted_indices = feature_importances.argsort()[::-1]

top_n = 10

# Tampilkan fitur-fitur penting
top_features = [x.columns[idx] for idx in sorted_indices[:top_n]]
other_features = [x.columns[idx] for idx in sorted_indices[top_n:]]

# Menggabungkan "Others" untuk fitur yang tidak termasuk dalam top 10
feature_labels = top_features + ['Others']
feature_importances = [feature_importances[idx] for idx in sorted_indices[:top_n]] + [sum(feature_importances[idx] for idx in sorted_indices[top_n:])]

# Visualisasikan fitur-fitur penting
plt.figure(figsize=(10, 6))
plt.barh(range(len(feature_labels)), feature_importances)
plt.yticks(range(len(feature_labels)), feature_labels)
plt.xlabel('Important Feature', color = 'darkblue')
plt.ylabel('Feature', color = 'darkblue')
plt.title('Important Feature from Decision Tree Model',  pad = 15, fontsize = 20, color = 'red')
plt.tight_layout()
plt.show()


# In[43]:


cust['avg amount'].mean()


# In[44]:


y_pred

