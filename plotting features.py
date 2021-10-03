%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import rcParams
# ^^^ pyforest auto-imports - don't write above this line
import scipy.io
import pandas as pd
import preprocessing_helper
import feature_extraction
# In[ ]:

directory_tim = 'D:\\Work\\Oxford\\Data\\TIM\\'
directory_acc = 'D:\\Work\\Oxford\\Data\\ACC\\'
directory_emg = 'D:\\Work\\Oxford\\Data\\EMG\\'
sampling_rate=1000
#extracting data
input_acc, envelope_acc, instantaneous_freq = preprocessing_helper.ThreeAxisACC(directory_acc,sampling_rate,sliding=False,seconds=1)
input_emg, envelope_emg = preprocessing_helper.ThreeAxisEMG(directory_emg,sampling_rate,sliding=False,seconds=1)
#extracting features
acc_feature_withoutenvelope = feature_extraction.featureACC(np.array(input_acc),cover= False,sampling_rate=sampling_rate)
acc_feature_withenvelope = feature_extraction.featureACC(np.array(envelope_acc),cover= True,sampling_rate=sampling_rate)
TSI = feature_extraction.TSI_feature(np.array(instantaneous_freq))
TSI = np.expand_dims(TSI, axis=2)
emg_feature_withoutenvelope = feature_extraction.featureEMG(np.array(input_emg),cover= True,sampling_rate=sampling_rate)
emg_feature_withenvelope = feature_extraction.featureEMG(np.array(envelope_emg),cover= False,sampling_rate=sampling_rate)

data_temp = np.concatenate((acc_feature_withoutenvelope, acc_feature_withenvelope, TSI,emg_feature_withoutenvelope,emg_feature_withenvelope), axis=2)

#For normalizing the values
for i in range(data_temp.shape[2]):
    data_temp[:,:,i,:] =  (data_temp[:,:,i,:] - data_temp[:,:,i,:].mean())/data_temp[:,:,i,:].std()

# taking one patient
temp = data_temp[0]#this has
rcParams['figure.figsize'] = 50, 10  # setting the size of plotting window
print(temp.shape)
aron = 0

# In[ ]:


ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=2, colspan=1)

ax1.hist(temp[0][aron], bins=10)  # x-axis
ax2.hist(temp[1][aron], bins=10)  # y-axis
ax3.hist(temp[2][aron], bins=10)  # z-axis
plt.xlabel("Power ACC")
plt.show()
aron += 1

# In[ ]:


ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=2, colspan=1)

ax1.hist(temp[0][aron], bins=10)  # x-axis
ax2.hist(temp[1][aron], bins=10)  # y-axis
ax3.hist(temp[2][aron], bins=10)  # z-axis
plt.xlabel("Energy ACC")
plt.show()
aron += 1

# In[ ]:


ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=2, colspan=1)

ax1.hist(temp[0][aron], bins=10)  # x-axis
ax2.hist(temp[1][aron], bins=10)  # y-axis
ax3.hist(temp[2][aron], bins=10)  # z-axis
plt.xlabel("Avg Val ACC")
plt.show()
aron += 1

# In[ ]:


ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=2, colspan=1)

ax1.hist(temp[0][aron], bins=10)  # x-axis
ax2.hist(temp[1][aron], bins=10)  # y-axis
ax3.hist(temp[2][aron], bins=10)  # z-axis
plt.xlabel("Variance ACC")
plt.show()
aron += 1

# In[ ]:


ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=2, colspan=1)

ax1.hist(temp[0][aron], bins=10)  # x-axis
ax2.hist(temp[1][aron], bins=10)  # y-axis
ax3.hist(temp[2][aron], bins=10)  # z-axis
plt.xlabel("First derivative ACC")
plt.show()
aron += 1

# In[ ]:


ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=2, colspan=1)

ax1.hist(temp[0][aron], bins=10)  # x-axis
ax2.hist(temp[1][aron], bins=10)  # y-axis
ax3.hist(temp[2][aron], bins=10)  # z-axis
plt.xlabel("DOM Freq ACC")
plt.show()
aron += 1

# In[ ]:


ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=2, colspan=1)

ax1.hist(temp[0][aron], bins=10)  # x-axis
ax2.hist(temp[1][aron], bins=10)  # y-axis
ax3.hist(temp[2][aron], bins=10)  # z-axis
plt.xlabel("Freq Range")
plt.show()
aron += 1

# In[ ]:


ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=2, colspan=1)

ax1.hist(temp[0][aron], bins=10)  # x-axis
ax2.hist(temp[1][aron], bins=10)  # y-axis
ax3.hist(temp[2][aron], bins=10)  # z-axis
plt.xlabel("Power ACCenv")
plt.show()
aron += 1

# In[ ]:


ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=2, colspan=1)

ax1.hist(temp[0][aron], bins=10)  # x-axis
ax2.hist(temp[1][aron], bins=10)  # y-axis
ax3.hist(temp[2][aron], bins=10)  # z-axis
plt.xlabel("Energy ACCenv")
plt.show()
aron += 1

# In[ ]:


ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=2, colspan=1)

ax1.hist(temp[0][aron], bins=10)  # x-axis
ax2.hist(temp[1][aron], bins=10)  # y-axis
ax3.hist(temp[2][aron], bins=10)  # z-axis
plt.xlabel("avg val ACCenv")
plt.show()
aron += 1

# In[ ]:


ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=2, colspan=1)

ax1.hist(temp[0][aron], bins=10)  # x-axis
ax2.hist(temp[1][aron], bins=10)  # y-axis
ax3.hist(temp[2][aron], bins=10)  # z-axis
plt.xlabel("variance ACCenv")
plt.show()
aron += 1

# In[ ]:


ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=2, colspan=1)

ax1.hist(temp[0][aron], bins=10)  # x-axis
ax2.hist(temp[1][aron], bins=10)  # y-axis
ax3.hist(temp[2][aron], bins=10)  # z-axis
plt.xlabel("first derivative ACCenv")
plt.show()
aron += 1

# In[ ]:


ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=2, colspan=1)

ax1.hist(temp[0][aron], bins=10)  # x-axis
ax2.hist(temp[1][aron], bins=10)  # y-axis
ax3.hist(temp[2][aron], bins=10)  # z-axis
plt.xlabel("dom freq ACCenv")
plt.show()
aron += 1

# In[ ]:


ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=2, colspan=1)

ax1.hist(temp[0][aron], bins=10)  # x-axis
ax2.hist(temp[1][aron], bins=10)  # y-axis
ax3.hist(temp[2][aron], bins=10)  # z-axis
plt.xlabel("freq range ACCenv")
plt.show()
aron += 1

# In[ ]:


ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=2, colspan=1)

ax1.hist(temp[0][aron], bins=10)  # x-axis
ax2.hist(temp[1][aron], bins=10)  # y-axis
ax3.hist(temp[2][aron], bins=10)  # z-axis
plt.xlabel("TSI")
plt.show()
aron += 1

# In[ ]:


ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=2, colspan=1)

ax1.hist(temp[0][aron], bins=10)  # x-axis
ax2.hist(temp[1][aron], bins=10)  # y-axis
ax3.hist(temp[2][aron], bins=10)  # z-axis
plt.xlabel("Power EMG")
plt.show()
aron += 1

# In[ ]:


ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=2, colspan=1)

ax1.hist(temp[0][aron], bins=10)  # x-axis
ax2.hist(temp[1][aron], bins=10)  # y-axis
ax3.hist(temp[2][aron], bins=10)  # z-axis
plt.xlabel("Energy EMG")
plt.show()
aron += 1

# In[ ]:


ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=2, colspan=1)

ax1.hist(temp[0][aron], bins=10)  # x-axis
ax2.hist(temp[1][aron], bins=10)  # y-axis
ax3.hist(temp[2][aron], bins=10)  # z-axis
plt.xlabel("Avg Val EMG")
plt.show()
aron += 1

# In[ ]:


ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=2, colspan=1)

ax1.hist(temp[0][aron], bins=10)  # x-axis
ax2.hist(temp[1][aron], bins=10)  # y-axis
ax3.hist(temp[2][aron], bins=10)  # z-axis
plt.xlabel("Variance EMG")
plt.show()
aron += 1

# In[ ]:


ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=2, colspan=1)

ax1.hist(temp[0][aron], bins=10)  # x-axis
ax2.hist(temp[1][aron], bins=10)  # y-axis
ax3.hist(temp[2][aron], bins=10)  # z-axis
plt.xlabel("first derivative EMG")
plt.show()
aron += 1

# In[ ]:


ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=2, colspan=1)

ax1.hist(temp[0][aron], bins=10)  # x-axis
ax2.hist(temp[1][aron], bins=10)  # y-axis
ax3.hist(temp[2][aron], bins=10)  # z-axis
plt.xlabel("Dom freq1 EMG")
plt.show()
aron += 1

# In[ ]:


ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=2, colspan=1)

ax1.hist(temp[0][aron], bins=10)  # x-axis
ax2.hist(temp[1][aron], bins=10)  # y-axis
ax3.hist(temp[2][aron], bins=10)  # z-axis
plt.xlabel("freq range1 EMG")
plt.show()
aron += 1

# In[ ]:


ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=2, colspan=1)

ax1.hist(temp[0][aron], bins=10)  # x-axis
ax2.hist(temp[1][aron], bins=10)  # y-axis
ax3.hist(temp[2][aron], bins=10)  # z-axis
plt.xlabel("Dom freq2 EMG")
plt.show()
aron += 1

# In[ ]:


ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=2, colspan=1)

ax1.hist(temp[0][aron], bins=10)  # x-axis
ax2.hist(temp[1][aron], bins=10)  # y-axis
ax3.hist(temp[2][aron], bins=10)  # z-axis
plt.xlabel("freq range2 EMG")
plt.show()
aron += 1

# In[ ]:


ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=2, colspan=1)

ax1.hist(temp[0][aron], bins=10)  # x-axis
ax2.hist(temp[1][aron], bins=10)  # y-axis
ax3.hist(temp[2][aron], bins=10)  # z-axis
plt.xlabel("Power EMGenv")
plt.show()
aron += 1

# In[ ]:


ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=2, colspan=1)

ax1.hist(temp[0][aron], bins=10)  # x-axis
ax2.hist(temp[1][aron], bins=10)  # y-axis
ax3.hist(temp[2][aron], bins=10)  # z-axis
plt.xlabel("Energy EMGenv")
plt.show()
aron += 1

# In[ ]:


ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=2, colspan=1)

ax1.hist(temp[0][aron], bins=10)  # x-axis
ax2.hist(temp[1][aron], bins=10)  # y-axis
ax3.hist(temp[2][aron], bins=10)  # z-axis
plt.xlabel("avg val EMGenv")
plt.show()
aron += 1

# In[ ]:


ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=2, colspan=1)

ax1.hist(temp[0][aron], bins=10)  # x-axis
ax2.hist(temp[1][aron], bins=10)  # y-axis
ax3.hist(temp[2][aron], bins=10)  # z-axis
plt.xlabel("variance EMGenv")
plt.show()
aron += 1

# In[ ]:


ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=2, colspan=1)

ax1.hist(temp[0][aron], bins=10)  # x-axis
ax2.hist(temp[1][aron], bins=10)  # y-axis
ax3.hist(temp[2][aron], bins=10)  # z-axis
plt.xlabel("first derivative EMGenv")
plt.show()
aron += 1

# In[ ]:


ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=2, colspan=1)

ax1.hist(temp[0][aron], bins=10)  # x-axis
ax2.hist(temp[1][aron], bins=10)  # y-axis
ax3.hist(temp[2][aron], bins=10)  # z-axis
plt.xlabel("dom freq EMGenv")
plt.show()
aron += 1

# In[ ]:


ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=2, colspan=1)
ax2 = plt.subplot2grid(shape=(6, 1), loc=(2, 0), rowspan=2, colspan=1)
ax3 = plt.subplot2grid(shape=(6, 1), loc=(4, 0), rowspan=2, colspan=1)

ax1.hist(temp[0][aron], bins=10)  # x-axis
ax2.hist(temp[1][aron], bins=10)  # y-axis
ax3.hist(temp[2][aron], bins=10)  # z-axis
plt.xlabel("freq range EMGenv")
plt.show()
aron += 1

# In[20]:


######################################DONE####################################################

