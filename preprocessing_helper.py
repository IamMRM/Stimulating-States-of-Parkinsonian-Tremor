import statistics
import numpy as np
import os
import scipy.io
import pandas as pd
import math
from scipy import signal
from scipy.signal import hilbert
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans, DBSCAN

def testing_preprocessing(name,fs):
    for i in range(len(name)):
        #print(len(name[i][0]))
        for j in range(len(name[i][0])):
            if len(name[i][0][j]) != int(fs*5):
                raise Exception("There is an error in preprocessing")
    print("Preprocessing worked")

def z_score(df):
    df_std = df.copy()# copy the dataframe
    for column in df_std.columns:# apply the z-score method
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
    return df_std

#JUST FOR PUTTING THE RANGE OF NUMBERS
def seq(start, stop, step=1):
    n = int(round((stop - start)/float(step)))
    if n > 1:
        return([start + step*i for i in range(n+1)])
    elif n == 1:
        return([start])
    else:
        return([])

def AllPersonAllaxis(acc_withoutEnv):
    variable = acc_withoutEnv#(9, 3, 31)
    fet = []
    final = []
    chek = np.zeros(shape=(variable.shape[0]*variable.shape[1]*variable.shape[3],variable.shape[2]))#1*3*75
    jehan=-1
    for i, axis in enumerate(variable):# every patient #(3, 31) ####9
        for j,fet in enumerate(axis):#(31,)  ####27 as 9*3
            temporar = np.zeros(shape=(fet.shape[0],len(fet[0])))#(31, 371)
            for k,inst in enumerate(fet):#(371,) ####837 as 9*3*31
                #print(len(inst))
                for l,abc in enumerate(inst):
                    temporar[k][l]=abc
            temporar=np.moveaxis(temporar,-1,0)#(371, 31)
            #temporar[np.where(temporar == 0)]
            #print(f"Number of Zeroes in Array --> {temporar[np.where(temporar == 0)].size}")
            #print(temporar.shape)
            for m,dua in enumerate(temporar):
                jehan+=1
                for n in range(dua.shape[0]):
                    chek[jehan,n] = dua[n]
    return chek

def FirstPersonFirstaxis(acc_withoutEnv):
    variable = acc_withoutEnv[0][0]
    chek = np.zeros(shape=((variable.shape[0]),len(variable[0])))
    #print(chek.shape)
    for i,data in enumerate(variable):
        chek[i]=np.array(data)
    return chek

def ACC_preprocessing_helper(df_cars, criterian,fs):
    temp_envelope = []
    temp_group_inst_freq = []
    df_cars_standardized = z_score(df_cars)
    analytic_signal = hilbert(df_cars_standardized[0])# hilbert
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0 * np.pi) * fs)
    instantaneous_frequency = np.pad(instantaneous_frequency, (0, 1), 'constant')

    ix,k,mx,ear = [],[],[],[]
    count = 0
    for abc, j in enumerate(df_cars_standardized[0]):
        k.append(j)
        mx.append(amplitude_envelope[abc])
        ear.append(instantaneous_frequency[abc])
        count += 1
        if count == criterian:
            temp_envelope.append(mx)
            temp_group_inst_freq.append(ear)
            ix.append(k)
            count = 0
            ear,mx,k = [],[],[]
    return ix, temp_envelope, temp_group_inst_freq

def sliding_ACC_preprocessing_helper(df_cars, criterian, fs, seconds):
    temp_envelope = []
    temp_group_inst_freq = []
    df_cars_standardized = z_score(df_cars)
    analytic_signal = hilbert(df_cars_standardized[0])  # hilbert
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0 * np.pi) * fs)
    instantaneous_frequency = np.pad(instantaneous_frequency, (0, 1), 'constant')

    ix, k, mx, ear = [], [], [], []
    imane = 0
    count = 0 + imane
    while (count != len(df_cars_standardized[0])):
        # for abc, j in enumerate(df_cars_standardized[0]):
        k.append(df_cars_standardized[0][count])
        mx.append(amplitude_envelope[count])
        ear.append(instantaneous_frequency[count])
        count += 1
        if count == criterian + imane:
            temp_envelope.append(mx)
            temp_group_inst_freq.append(ear)
            ix.append(k)

            imane += fs * seconds
            count = imane
            # print("The starting is ",str(count))
            # print("**********")
            ear, mx, k = [], [], []
        # else:
        # print("BLA ",str(count))
    return ix, temp_envelope, temp_group_inst_freq

def ThreeAxisACC(directory_acc,sampling_freq,sliding=False,seconds=1):
    patient = []
    input_images = []
    envelope = []
    envelope_group = []

    fs = sampling_freq
    temp_instantaneous_freq = []
    instantaneous_freq = []

    for index_patient, filename in enumerate(os.listdir(directory_acc)):
        print(filename)
        mat = scipy.io.loadmat(directory_acc + filename)
        df_cars = pd.DataFrame(mat['tremorxf'][0, :])
        if sliding:
            ix, temp_envelope, temp_group_inst_freq = sliding_ACC_preprocessing_helper(df_cars,int(fs*5),fs,seconds)
        else:
            ix, temp_envelope, temp_group_inst_freq = ACC_preprocessing_helper(df_cars,int(fs*5),fs)
        patient.append(ix)
        envelope.append(temp_envelope)
        temp_instantaneous_freq.append(temp_group_inst_freq)
        temp_envelope, temp_group_inst_freq = [], []

        df_cars = pd.DataFrame(mat['tremoryf'][0, :])
        if sliding:
            iy, temp_envelope, temp_group_inst_freq = sliding_ACC_preprocessing_helper(df_cars,int(fs*5),fs,seconds)
        else:
            iy, temp_envelope, temp_group_inst_freq = ACC_preprocessing_helper(df_cars,int(fs*5),fs)
        patient.append(iy)
        envelope.append(temp_envelope)
        temp_instantaneous_freq.append(temp_group_inst_freq)
        temp_envelope, temp_group_inst_freq = [], []

        df_cars = pd.DataFrame(mat['tremorzf'][0, :])
        if sliding:
            iz, temp_envelope, temp_group_inst_freq = sliding_ACC_preprocessing_helper(df_cars,int(fs*5),fs,seconds)
        else:
            iz, temp_envelope, temp_group_inst_freq = ACC_preprocessing_helper(df_cars,int(fs*5),fs)
        patient.append(iz)
        envelope.append(temp_envelope)
        temp_instantaneous_freq.append(temp_group_inst_freq)
        temp_envelope, temp_group_inst_freq = [], []

        input_images.append(patient)
        envelope_group.append(envelope)
        instantaneous_freq.append(temp_instantaneous_freq)
        patient, envelope, temp_instantaneous_freq = [], [], []

    print("Done")
    testing_preprocessing(input_images,fs)
    testing_preprocessing(envelope_group,fs)
    testing_preprocessing(instantaneous_freq,fs)

    return input_images, envelope_group, instantaneous_freq

def OneAxisACC(directory_acc,sampling_freq): # here the data should have 'X'
    patient = []
    input_images = []
    envelope = []
    envelope_group = []

    fs = sampling_freq
    temp_instantaneous_freq = []
    instantaneous_freq = []

    for index_patient, filename in enumerate(os.listdir(directory_acc)):
        print(filename)
        mat = scipy.io.loadmat(directory_acc + filename)
        df_cars = pd.DataFrame(mat['X'][1, :])
        ix, temp_envelope, temp_group_inst_freq = ACC_preprocessing_helper(df_cars,int(fs*5),fs)
        patient.append(ix)
        envelope.append(temp_envelope)
        temp_instantaneous_freq.append(temp_group_inst_freq)
        temp_envelope, temp_group_inst_freq = [], []

        input_images.append(patient)
        envelope_group.append(envelope)
        instantaneous_freq.append(temp_instantaneous_freq)
        patient, envelope, temp_instantaneous_freq = [], [], []

    print("Done")
    testing_preprocessing(input_images,fs)
    testing_preprocessing(envelope_group,fs)
    testing_preprocessing(instantaneous_freq,fs)

    return input_images, envelope_group, instantaneous_freq

def butter_highpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=2):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=2):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def sliding_EMG_preprocessing_helper(sine, criterian, fs, seconds):
    temp_envelope = []
    filtered_sine = butter_highpass_filter(sine.data, 15, fs)
    beatriz = butter_lowpass_filter(abs(filtered_sine), 150, fs)
    total = np.sum(beatriz)
    normal_array = beatriz / total
    # print(np.sum(normal_array))
    df_cars = pd.DataFrame(normal_array)
    analytic_signal = hilbert(df_cars[0])
    amplitude_envelope = np.abs(analytic_signal)

    ix, k, mx = [], [], []
    imane = 0
    count = 0 + imane
    while (count != len(df_cars[0])):
        k.append(df_cars[0][count])
        mx.append(amplitude_envelope[count])
        count += 1
        if count == criterian + imane:
            temp_envelope.append(mx)
            ix.append(k)
            imane += fs * seconds
            count = imane
            mx, k = [], []
    return ix, temp_envelope

def EMG_preprocessing_helper(sine, criterian, fs):
    temp_envelope = []
    filtered_sine = butter_highpass_filter(sine.data, 15, fs)
    beatriz = butter_lowpass_filter(abs(filtered_sine), 150, fs)
    total = np.sum(beatriz)
    normal_array = beatriz / total
    # print(np.sum(normal_array))
    df_cars = pd.DataFrame(normal_array)
    analytic_signal = hilbert(df_cars[0])
    amplitude_envelope = np.abs(analytic_signal)

    ix, k, mx = [], [], []
    count = 0
    for abc, j in enumerate(df_cars[0]):
        k.append(j)
        mx.append(amplitude_envelope[abc])
        count += 1
        if count == criterian:
            temp_envelope.append(mx)
            ix.append(k)
            count = 0
            mx, k = [], []
    return ix, temp_envelope

def ThreeAxisEMG(directory_emg, sampling_freq, sliding=False, seconds=1):
    order = 2
    fs = sampling_freq
    cutoff_freq = 15
    print("Order of the filter is ", order)
    print("The cutoff frequency is ", cutoff_freq)
    print("After Nyquist is ", cutoff_freq / (0.5 * fs))

    input_emg = []
    patient = []
    temp_envelope = []
    envelope = []
    envelope_group = []

    for index_patient, filename in enumerate(os.listdir(directory_emg)):
        print(filename)
        mat = scipy.io.loadmat(directory_emg + filename)
        # x-AXIS
        sine = mat['mio1fx'][0, :]
        if sliding:
            ix, temp_envelope = sliding_EMG_preprocessing_helper(sine, int(fs * 5), fs, seconds)
        else:
            ix, temp_envelope = EMG_preprocessing_helper(sine, int(fs * 5), fs)
        patient.append(ix)
        envelope.append(temp_envelope)
        temp_envelope = []

        # y-axis
        sine = mat['mio1fy'][0, :]
        if sliding:
            iy, temp_envelope = sliding_EMG_preprocessing_helper(sine, int(fs * 5), fs, seconds)
        else:
            iy, temp_envelope = EMG_preprocessing_helper(sine, int(fs * 5), fs)
        patient.append(iy)
        envelope.append(temp_envelope)
        temp_envelope = []

        # z-axis
        sine = mat['mio1fz'][0, :]
        if sliding:
            iz, temp_envelope = sliding_EMG_preprocessing_helper(sine, int(fs * 5), fs, seconds)
        else:
            iz, temp_envelope = EMG_preprocessing_helper(sine, int(fs * 5), fs)
        patient.append(iz)
        envelope.append(temp_envelope)
        temp_envelope = []

        input_emg.append(patient)
        envelope_group.append(envelope)
        patient, envelope = [], []

    print("Done EMG")
    testing_preprocessing(input_emg, fs)
    testing_preprocessing(envelope_group, fs)

    return input_emg, envelope_group

def optimal_cluster_value(norm_matrix):
    min_value=-999.99
    if norm_matrix.shape[0]-1 > 10:
        range_n_clusters = seq(2,10, 1)
    else:
        range_n_clusters = seq(2, norm_matrix.shape[0] - 1, 1)

    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters,random_state=10)#random_state makes the results reproducible and can be useful for debugging
        cluster_labels = clusterer.fit_predict(norm_matrix)
        silhouette_avg = silhouette_score(norm_matrix,cluster_labels)
        print("For n clusters = ",n_clusters)
        print("The avg silhoute score is : ",silhouette_avg)
        if min_value<silhouette_avg:
            min_value=silhouette_avg
            value=n_clusters
        #computing the silhoute scores for each sample
        sample_silhoute_values = silhouette_samples(norm_matrix,cluster_labels)
    print("According to KMeans the cluster number is ",str(value))
    return value

def optimal_eps_value(norm_matrix,n_clusters):
    range_eps = seq(0.001, 9, 0.0001)
    cash = True
    max_value = 0.0
    eps_finalVal = 0.0
    for i in range_eps:
        db = DBSCAN(eps=i, min_samples=n_clusters).fit(neu)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        unique, counts = np.unique(labels, return_counts=True)
        # print(unique)
        if len(set(labels)) > 2:
            silhouette_avg = silhouette_score(neu, labels)

            # print("The average silhouette_score is : ",silhouette_avg,"For eps value = "+str(i))
            if silhouette_avg > 0:
                if len(unique) > 2:
                    print("The average silhouette_score is : ", silhouette_avg, "For eps value = " + str(i))
                    print(dict(zip(unique, counts)))
                    if max_value < silhouette_avg:
                        max_value = silhouette_avg
                        eps_finalVal = i
                    cash = False
    print("The best eps ", eps_finalVal)
    return eps_finalVal


"""def optimal_eps_value(norm_matrix,n_clusters):
    range_eps = seq(0.001, 9, 0.0001)
    cash=True
    max_value= 0.0
    eps_finalVal=0.0
    for i in range_eps:
        db = DBSCAN(eps=i,min_samples=n_clusters).fit(norm_matrix)
        core_samples_mask = np.zeros_like(db.labels_,dtype=bool)
        core_samples_mask[db.core_sample_indices_]=True
        labels = db.labels_
        #print(set(labels))
        unique, counts = np.unique(labels, return_counts=True)
        try:
            silhouette_avg = silhouette_score(norm_matrix, labels)
            if silhouette_avg > 0:
                if len(unique) > 2:
                    print("The average silhouette_score is : ",silhouette_avg,"For eps value = "+str(i))
                    print(dict(zip(unique, counts)))
                    if max_value<silhouette_avg:
                        max_value=silhouette_avg
                        eps_finalVal=i
                    cash= False
        except:
            if cash:
                continue
            else:
                break
    print("The best eps ",eps_finalVal)
    return eps_finalVal"""

def DB_cluster(norm_matrix,eps_finalVal,cluster_final):
    #min_samples= seq(2,norm_matrix.shape[0]-1,1)
    min_samples = seq(2, 10, 1)
    maxOutliers=999
    for i in min_samples:
        db = DBSCAN(eps=eps_finalVal,min_samples=i).fit(norm_matrix)
        core_samples_mask = np.zeros_like(db.labels_,dtype=bool)
        core_samples_mask[db.core_sample_indices_]=True
        count_labels = set([label for label in db.labels_ if label >=0])
        labels=db.labels_
        unique, counts = np.unique(labels, return_counts=True)
        print(dict(zip(unique, counts)))
        print("Total no. of clusters are ",str(len(set(count_labels))))
        if len(set(count_labels)) == cluster_final:
            if dict(zip(unique, counts))[-1] < maxOutliers:
                maxOutliers=dict(zip(unique, counts))[-1]
                final_dist=i
    print("SO the final value for clusters is ",final_dist)
    return final_dist

def nnormalize(chek):  # shape Feature x instances
    minimum = chek.min(axis=1)
    mimim = abs(np.array([minimum, ] * chek.shape[1]).transpose())
    chek += mimim
    # Checking for -ve values (CHECKED)
    if np.any(chek < 0):
        print("We have negative values in the original array")
    # checking for NAN values
    array_sum = np.sum(chek)
    array_has_nan = np.isnan(array_sum)
    if array_has_nan.any():
        print("NaN value detected in the original array")
    # Normalizing the values
    total = np.sum(chek, axis=1)  # total.shape = (features, )
    total_array = np.array([total, ] * chek.shape[1]).transpose()
    print(total_array.shape)

    # Checking for 0 value (CHECKED)
    if np.any(total_array == 0.0):
        print("It contains 0 values which is not suitable for division so NORMALIZATION FAILED")
    else:
        norm_matrix = chek / total_array
        print("NORMALIZATION SUCCESS")
    # CHECKING THE NORMALIZATION SUM AND NEGATIVE VALUES
    # print(norm_matrix.shape)
    for i in range(norm_matrix.shape[0]):
        if round(np.sum(norm_matrix[i, :])) == 1.0:
            continue
        else:
            print("Sum is not coming out to be 1 for ", i)
    if np.any(norm_matrix < 0):
        print("We have negative values in the normalized array")
    # checking for NAN values
    array_sum = np.sum(norm_matrix)
    array_has_nan = np.isnan(array_sum)
    if array_has_nan.any():
        print("NaN value detected in the normalized array")

    return norm_matrix

def downscale(magic):
    #print(f"magic is {magic}")
    a = int(magic.shape[0]/2000)# 20
    #print(f"magic2 is {a}")
    temporary= np.zeros(shape=(a))
    #print(f"magic3 is {temporary}")
    vari=0
    for i in range(a):# 0 till 19
        lst = magic[vari:2000*(i+1)]
        values, counts = np.unique(lst, return_counts=True)
        #print(dict(zip(values,counts)))
        ind = np.argmax(counts)
        #print(values[ind])
        #print(f"magic4 is {lst}")
        #temporary[i]=max(lst,key=lst.count)
        #print(lst.shape)
        #counts = np.bincount(lst)
        temporary[i]=values[ind]#np.argmax(counts)
        #print(values[ind])
        #print(f"magic5 is {temporary[i]}")
        vari=2000*(i+1)
        #print(f"magic6 is {vari}")
    #print(f"magic7 is {temporary}")
    return temporary

def show_time_clusters(X,Y):
    plt.plot(X)#,"r.")
    plt.plot(Y)
    #plt.plot(a)
    plt.ylabel("cluster labels")
    plt.xlabel("Time points")
    plt.show()

def time_plot_Val(directory_tim, ):
    var= 0
    for index_patient,filename in enumerate(os.listdir(directory_tim)):
        print(filename)
        mat = scipy.io.loadmat(directory_tim+filename)
        label=pd.DataFrame(mat['X'][2,:])
        x= pd.DataFrame(mat['X'][1,:])
        clus = downscale(label)
        show_time_clusters(cluster[var:20*(index_patient+1)],clus)
        var=20*(index_patient+1)