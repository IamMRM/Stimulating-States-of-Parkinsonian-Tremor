import matplotlib.pyplot as plt
# ^^^ pyforest auto-imports - don't write above this line
import scipy.io
import pandas as pd
import preprocessing_helper
import feature_extraction
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

directory_validation = 'D:\\Work\\Oxford\\Data\\TIM\\'
directory_acc = 'D:\\Work\\Oxford\\Data\\ACC\\'
directory_emg = 'D:\\Work\\Oxford\\Data\\EMG\\'

sampling_rate=1000


# In[4]:

#Here for having overlapping regions do sliding = True. And if you want to have 4 sec sliding then seconds = 1
input_acc, envelope_acc, instantaneous_freq = preprocessing_helper.ThreeAxisACC(directory_acc,sampling_rate,sliding=False,seconds=1)
input_emg, envelope_emg = preprocessing_helper.ThreeAxisEMG(directory_emg,sampling_rate,sliding=False,seconds=1)


# In[6]:

#incase you want to save the dataset
"""np.save("accS",arr = np.array(input_acc))
np.save("acc_envelopeS",arr = np.array(envelope_acc))
np.save("acc_instantaneous_freqS",arr = np.array(instantaneous_freq))
np.save("emgS",arr = np.array(input_emg))
np.save("emg_envelopeS",arr = np.array(envelope_emg))"""
#sys.modules[__name__].__dict__.clear()


# In[9]:

#Calculating FEATURES
acc_feature_withoutenvelope = feature_extraction.featureACC(np.array(input_acc),cover= False,sampling_rate=sampling_rate)
acc_feature_withenvelope = feature_extraction.featureACC(np.array(envelope_acc),cover= True,sampling_rate=sampling_rate)
TSI = feature_extraction.TSI_feature(np.array(instantaneous_freq))
TSI = np.expand_dims(TSI, axis=2)
emg_feature_withoutenvelope = feature_extraction.featureEMG(np.array(input_emg),cover= True,sampling_rate=sampling_rate)
emg_feature_withenvelope = feature_extraction.featureEMG(np.array(envelope_emg),cover= False,sampling_rate=sampling_rate)

# In[14]:

print(acc_feature_withoutenvelope.shape)
print(acc_feature_withenvelope.shape)
print(TSI.shape)
print(emg_feature_withoutenvelope.shape)
print(emg_feature_withenvelope.shape)

# In[15]:

data_temp = np.concatenate((acc_feature_withoutenvelope, acc_feature_withenvelope, TSI,emg_feature_withoutenvelope,emg_feature_withenvelope), axis=2)
print(data_temp.shape)
print(data_temp.shape[0])#patient
print(data_temp.shape[1])#axis
print(data_temp.shape[2])#features
print(data_temp.shape[3])#instances

#Z-Normalization
for i in range(data_temp.shape[2]):
    data_temp[:,:,i,:] =  (data_temp[:,:,i,:] - data_temp[:,:,i,:].mean())/data_temp[:,:,i,:].std()
#Checking
print(data_temp[:,:,0,:].mean())
print(data_temp[:,:,0,:].std())

# In[21]:

print(data_temp.shape)
#now have to convert into shape=(instances x features)
print(data_temp.shape[0]*data_temp.shape[1]*data_temp.shape[3])

neu = preprocessing_helper.AllPersonAllaxis(data_temp)
print(neu.shape)
def show_data(X):
    plt.plot(X)#,"r.")
    #plt.plot(a)
    plt.ylabel("Data values")
    plt.xlabel("Instances")
    plt.show()
show_data(neu)

# In[23]:

#Finding optimal parameters for clustering
n_clusters = preprocessing_helper.optimal_cluster_value(neu)
eps_finalVal = preprocessing.optimal_eps_value(neu,n_clusters)
final_cluster = preprocessing_helper.DB_cluster(neu,eps_finalVal,n_clusters)

# In[30]:

clustering = DBSCAN(eps=eps_finalVal, min_samples=final_cluster).fit(neu)
cluster=clustering.labels_
print(len(set(cluster)))
unique, counts = np.unique(cluster, return_counts=True)
dict(zip(unique, counts))


# In[32]:


"""def show_clusters(X,cluster):
    df=pd.DataFrame(dict(x=X[:,0],y=X[:,1], label=cluster))
    colors = {-1:'red',0:'blue',1:'orange',2:'green',3:'skyblue',4:'black'}
    fig,ax=plt.subplots(figsize=(8,8))
    grouped = df.groupby('label')
    
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter',x='x',y='y',label=key ,color=colors[key])
    plt.xlabel("feature 0")
    plt.ylabel("feature 1")
    plt.show()
show_clusters(neu,cluster)"""


# In[35]:

#PCA for plotting
# as PCA maximizes variance so normalization is a must
pca = PCA(.95)
#pca= PCA(n_components=2)
pca.fit(neu)#shape (n_samples, n_features)
train_img = pca.transform(neu)

# In[34]:

print(train_img.shape)
print(pca.components_.shape)

#Variance and important components
print(pca.explained_variance_ratio_)
n_pcs= pca.components_.shape[0]
most_important = [np.abs(pca.components_[:][i]).argmax() for i in range(n_pcs)]
print(most_important)

# In[37]:

"""# number of components
n_pcs= pca.components_.shape[0]

# get the index of the most important feature on EACH component i.e. largest absolute value
# using LIST COMPREHENSION HERE
most_important = [np.abs(pca.components_[:][i]).argmax() for i in range(n_pcs)]
initial_feature_names = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30']

# get the names
most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

# using LIST COMPREHENSION HERE AGAIN
dic = {'PC{}'.format(i+1): most_important_names[i] for i in range(n_pcs)}

# build the dataframe
df = pd.DataFrame(sorted(dic.items()))
print(df)"""


# In[38]:


def show_clusters(X,cluster):
    df=pd.DataFrame(dict(x=X[:,0],y=X[:,1], label=cluster))
    colors = {-1:'red',0:'blue',1:'orange',2:'green',3:'skyblue',4:'black',5:'purple'}
    fig,ax=plt.subplots(figsize=(8,8))
    grouped = df.groupby('label')
    
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter',x='x',y='y',label=key ,color=colors[key])
    plt.xlabel("feature 0")
    plt.ylabel("feature 1")
    plt.show()
show_clusters(train_img,cluster)