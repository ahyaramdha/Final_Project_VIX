#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install "numpy>=1.16.5,<1.23.0"


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import preprocessing

import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import StandardScaler
import sklearn
from sklearn import preprocessing
# To perform KMeans clustering 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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


custcount = df['CustomerID'].value_counts().reset_index()
custcount


# ingin memisahkan kustomer yang loyal (datang tiap bulannya/lebih dari sama dengan 12) dan tidak loyal

# In[9]:


# 1 loyal, 0 tidak loyal
def loyal(x):
    if x >= 12:
        return 1
    else :
        return 0

custcount['loyal'] = custcount['CustomerID'].apply(loyal)
custcount


# In[10]:


custcount = custcount.rename(columns = {'CustomerID': 'kunjungan', 'index':'CustomerID'})
custcount


# In[11]:


groupT = df.groupby(['CustomerID'])['TotalAmount'].median().reset_index()
groupT


# In[12]:


grouprod = df.groupby(['CustomerID'])['Product Name'].value_counts().unstack().reset_index().fillna(0)
grouprod


# In[13]:


cust = custcount.merge(df1, on='CustomerID').merge(grouprod, on = 'CustomerID').merge(groupT, on = 'CustomerID')
cust


# In[14]:


cust = cust.rename(columns = {'TotalAmount':'avg amount'})


# In[15]:


cust.columns


# In[16]:


cust['buytot'] = cust['Cashew'] + cust['Cheese Stick'] + cust['Coffee Candy'] + cust['Crackers'] + cust['Ginger Candy '] + cust['Oat'] + cust['Potato Chip'] + cust['Thai Tea'] + cust['Yoghurt']
cust


# disini kita ingin melihat bagaimana perilaku/ciri kustomer yang loyal dan tidak loyal

# In[17]:


status_onehot = pd.get_dummies(cust["Marital Status"], prefix= '_')
cust = cust.join(status_onehot)
cust.head(2)


# In[18]:


cust['Income'] = cust['Income'].str.replace(',', '.')
cust.head(2)


# In[19]:


cust['Income'] = cust['Income'].astype('float')
cust['Income'] = cust['Income']*1000000
cust.dtypes


# # Handle Outlier

# In[20]:


ax = sns.boxplot(data = cust['avg amount'])


# In[21]:


ax3 = sns.boxplot(data = cust['Income'])


# In[22]:


ax4 = sns.boxplot(data = cust['buytot'])


# In[23]:


#making function to replace outlier
def outlier (x):
    sorted(x)
    q1, q3 = x.quantile([0.25, 0.75])
    IQR = q3 - q1
    lwr_bound = q1 - (1.5*IQR)
    upr_bound = q3 + (1.5*IQR)
    return lwr_bound, upr_bound


# In[24]:


low, high = outlier(cust['Income'])
low2, high2 = outlier(cust['buytot'])
low3, high3 = outlier(cust['avg amount'])


# In[25]:


#replacing outlier with upper bound and lower bound value
cust['Income'] = np.where(cust['Income']>high, high, cust['Income'])
cust['Income'] = np.where(cust['Income']<low, low, cust['Income'])
cust['buytot'] = np.where(cust['buytot']>high2, high2, cust['buytot'])
cust['buytot'] = np.where(cust['buytot']<low2, low2, cust['buytot'])
cust['avg amount'] = np.where(cust['avg amount']>high3, high3, cust['avg amount'])
cust['avg amount'] = np.where(cust['avg amount']<low3, low3, cust['avg amount'])


# In[26]:


ax3 = sns.boxplot(data = cust['Income'])


# In[27]:


cust2 = cust


# In[28]:


cust = cust.drop(columns ={'CustomerID', 'kunjungan', 'loyal', 'Marital Status'}, axis = 1)
cust


# In[29]:


cust3 = cust


# # KMEANS

# In[30]:


scaler = StandardScaler()


# In[31]:


# Method 2: Modify the existing DataFrame in place
cust[cust.columns] = scaler.fit_transform(cust[cust.columns])


# In[32]:


cust


# In[74]:


# membuat fungsi Elbow Method
sns.set_theme('notebook', style='white')
def elbowMethod(cust, k_min=2, k_max= 20):
    wcss = [] # Within Cluster Sum of Squares
    k_range = range(k_min, k_max + 1)

    for i in k_range:
      kmeans_test = KMeans(n_clusters = i, random_state = 42, init = 'k-means++')
      kmeans_test.fit(cust)
      wcss.append(kmeans_test.inertia_)

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(k_range, wcss, marker='o')

    plt.axvline(x = 4, color = 'r')
    plt.title('The Elbow Method', fontsize = 13, color = 'red')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    
elbowMethod(cust)


# In[ ]:





# ### Melakukan Clustering

# ditentukan kalau jumlah cluster adalah 5

# In[58]:


kmeans = KMeans(n_clusters = 4, random_state = 42).fit(cust)
labels = kmeans.labels_


# In[60]:


score = []

for cluster in range(1,10):
    kmeans = KMeans(n_clusters = 4, init="k-means++", random_state=42)
    kmeans.fit(cust)
    score.append(kmeans.inertia_)

print('n-cluster = 4')
print()
hasilcl = cust.copy()
hasilcl['kmeans_4cluster'] = labels
print('Cluster and its customers quantity :')
display(hasilcl.kmeans_4cluster.value_counts(ascending=True))
display(hasilcl)


# In[61]:


cust2 = cust2.join(hasilcl['kmeans_4cluster'])
cust2


# In[62]:


cust2.columns


# In[69]:


num = cust2[['kunjungan', 'Age', 'Income', 'Cashew', 'Cheese Stick', 
             'Choco Bar', 'Coffee Candy', 'Crackers', 'Ginger Candy ', 'Oat', 
             'Potato Chip', 'Thai Tea', 'Yoghurt', 'avg amount', 'buytot',
             'kmeans_4cluster']]
nonum = cust2[['loyal', 'Gender', 'Marital Status', 'kmeans_4cluster']]


# In[70]:


#mencari modus untuk non numeric data
nonumgroup = nonum.groupby('kmeans_4cluster').apply(lambda x: x.mode().iloc[0])
nonumgroup


# In[71]:


numgroup = num.groupby('kmeans_4cluster').mean()
numgroup


# In[72]:


colors = plt.cm.tab20.colors


# In[75]:


for col in numgroup.columns[0:]:
    plt.figure()
    x_labels = numgroup[col].index
    x_pos = range(len(x_labels))
    
    plt.bar(x_pos, numgroup[col], color=colors)

    plt.title(f'Column {col}', fontsize=17, pad=15, color='darkred')
    plt.xlabel('Kelompok')
    plt.ylabel('Rata-rata')
    
    # Customize the x-axis tick labels
    plt.xticks(x_pos, x_labels, rotation=45, ha='right')
    
    plt.tight_layout()  # To make sure the labels fit properly
    plt.show()


# + **Kelompok 1:** merupakan kustomer loyal. Memiliki income dan rata-rata umur yang tinggi (paling tinggi) dan umumnya sudah menikah. Memiliki rata-rata konsumsi yang tinggi dengan rata-rata pembayaran yang sedang. didominasi oleh konsumen laki-laki.
# + **Kelompok 2:** bukanlah kustomer loyal. Memiliki income dan rata-rata umur yang sedang. Umumnya sudah menikah. Memiliki rata-rata konsumsi yang sedang, rata-rata pembayaran yang rendah, dan lebih sering membeli produk thai tea dan ginger candy. didominasi oleh konsumen laki-laki.
# + **Kelompok 3:** bukanlah kustomer loyal. Memiliki income dan rata-rata umur yang tinggi. Umumnya sudah menikah. Memiliki rata-rata pembayaran yang tinggi, namun jumlah konsumsi yang rendah (sepertinya lebih banyak membeli produk berharga tinggi) dan lebih sering membeli yoghurt dan choco bar. didominasi oleh konsumen laki-laki.
# + **Kelompok 4:** bukanlah kustomer loyal, namun memiliki jumlah kunjungan lebih dari 10 kali dalam 1 tahun. Memiliki income sedang dan rata-rata umur yang rendah. Umumnya masih single. Memiliki rata-rata pembayaran dan konsumsi yang sedang dan lebih sering membeli ginger candy (tidak sebanyak 1 dan 2). didominasi oleh konsumen laki-laki.
