import matplotlib.pyplot as plt
# ^^^ pyforest auto-imports - don't write above this line
import numpy as np
import scipy.io
from scipy.fftpack import fft
from scipy import signal
from scipy.signal import hilbert
from matplotlib import rcParams

def TSI_feature(acc_instFreq):
    TSI = list()
    axi = list()
    ins = list()

    for patientt in range(acc_instFreq.shape[0]):  # 10
        for axiss in range(acc_instFreq.shape[1]):  # 3
            for instt in range(np.array(acc_instFreq[patientt][axiss]).shape[0]):  # 75
                delta_f = np.diff(np.array(acc_instFreq[patientt][axiss])[instt])
                delta_f = np.pad(delta_f, (0, 1), 'constant')
                f = np.array(acc_instFreq[patientt][axiss])[instt]
                q75, q25 = np.percentile(delta_f, [75, 25])
                iqr = q75 - q25
                ins.append(iqr)
            axi.append(ins)
            ins = list()
        TSI.append(axi)
        axi = list()
    return np.array(TSI)

def dom_freq_with_range(power_spectrum1, frequency):
    dom_family1 = frequency[np.argmax(power_spectrum1)]
    di = np.diff(power_spectrum1)
    di = np.pad(di, (0, 1), 'constant')  # adds 0 at the end
    fre = []
    for tempindex, i in enumerate(di):  # as 2500 so every 5 point is 1
        if tempindex > (int(dom_family1) * 5) - 10 and tempindex < (int(dom_family1) * 5) + 10:  # means +-2
            if abs(i) >= max(power_spectrum1) - max(power_spectrum1) * .70:
                fre.append(frequency[tempindex])
    if len(fre) >= 2:
        freq = (abs(abs(fre[0]) - abs(fre[-1])))
    else:
        freq = 0.0
        print("There is no range(0) in this time instant")
    return dom_family1, freq

def featureACC(data, cover, sampling_rate):
    #                                      ACCELEROMETER RECORDINGS
    cover = cover
    power = []
    energy = []
    avgval = []
    varience = []
    first_derivate = []
    sampling_rate = sampling_rate  # 1000Hz
    dominantfreq = []
    freqrange = []

    features = []
    axi = []
    pa = []

    for pat in data:  # per patient
        for axis in pat:  # per axis
            e = 0.0
            var = 0.0
            for inst in axis:
                #################################
                fourier_transform = np.fft.rfft(inst)  # one dimensional discrete fourier transform
                abs_fourier_transform = np.abs(fourier_transform)
                power_spectrum = np.square(abs_fourier_transform)
                if cover:
                    power_spectrum[:10] = 0  # 0-2 are = 0 now
                frequency = np.linspace(0, sampling_rate / 2, len(power_spectrum))

                dom_family1, freqrang1 = dom_freq_with_range(power_spectrum, frequency)
                dominantfreq.append(dom_family1)
                freqrange.append(float(freqrang1))
                """
                dominantfreq.append(frequency[np.argmax(power_spectrum)])
                fre = []
                for tempindex, i in enumerate(np.diff(power_spectrum)):
                    if abs(i) >= 500:
                        fre.append(frequency[tempindex])
                if len(fre) >= 2:
                    freqrange.append(abs(abs(fre[0]) - abs(fre[-1])))
                else:
                    freqrange.append(0)"""
                #################################
                first_derivate.append(sum(np.diff(inst)) / len(np.diff(inst)))
                averg_val = sum(inst) / len(inst)  # har 5000 ka mean le rha h
                for val in inst:
                    e += float(val) * float(val)  # energy [J = J/s * 5 second]
                    var += (float(averg_val) - float(val)) * (float(averg_val) - float(val))
                p = e / (2 * len(inst))  # power over each window [J/s]
                var /= (len(inst) - 1)
                power.append(p)
                energy.append(e)
                avgval.append(averg_val)
                varience.append(var)
            features.append(power)
            features.append(energy)
            features.append(avgval)
            features.append(varience)
            features.append(first_derivate)
            features.append(dominantfreq)
            features.append(freqrange)
            axi.append(features)

            power = []
            energy = []
            avgval = []
            varience = []
            first_derivate = []
            dominantfreq = []
            freqrange = []
            features = []
        pa.append(axi)
        axi = []
    return np.array(pa)

def featureEMG(data, cover,sampling_rate):
    #                                          EMG RECORDINGS
    cover=cover
    power=[]
    energy=[]
    avgval=[]
    varience=[]
    first_derivate=[]
    sampling_rate = sampling_rate
    dominantfreq=[]
    freqrange=[]

    if cover:
        dominantfreq2=[]
        freqrange2=[]
    features=[]
    axi=[]
    pa=[]

    for pat in data:#per patient
        for axis in pat:#per axis
            e=0.0
            var=0.0
            for inst in axis:
                #################################
                fourier_transform = np.fft.rfft(inst)#one dimensional discrete fourier transform
                abs_fourier_transform = np.abs(fourier_transform)

                power_spectrum1 = np.square(abs_fourier_transform)
                power_spectrum1[:10]=0 # 2 to 14
                power_spectrum1[70:]=0
                frequency = np.linspace(0, sampling_rate/2, len(power_spectrum1))

                dom_family1, freqrang1= dom_freq_with_range(power_spectrum1,frequency)
                dominantfreq.append(dom_family1)
                freqrange.append(float(freqrang1))

                #PART 2
                if cover:
                    power_spectrum2 = np.square(abs_fourier_transform)
                    power_spectrum2[:70]=0 # 14 to 98
                    power_spectrum2[240:260]=0# 48 to 52 = 0
                    power_spectrum2[490:]=0 # beyond 98 = 0
                    frequency = np.linspace(0, sampling_rate/2, len(power_spectrum2))
                    dom_family2, freqrang2= dom_freq_with_range(power_spectrum2,frequency)
                    dominantfreq2.append(dom_family2)
                    freqrange2.append(float(freqrang2))
                #################################
                first_derivate.append(sum(np.diff(inst))/len(np.diff(inst)))
                averg_val = sum(inst)/len(inst)# har 5000 ka mean le rha h
                for val in inst:
                    e +=float(val)*float(val)#energy [J = J/s * 5 second]
                    var +=(float(averg_val)-float(val))*(float(averg_val)-float(val))
                p=e/(2*len(inst))#power over each window [J/s]
                var/=(len(inst)-1)
                power.append(p)
                energy.append(e)
                avgval.append(averg_val)
                varience.append(var)
            features.append(power)
            features.append(energy)
            features.append(avgval)
            features.append(varience)
            features.append(first_derivate)
            features.append(dominantfreq)
            features.append(freqrange)
            if cover:
                features.append(dominantfreq2)
                features.append(freqrange2)
            axi.append(features)

            power=[]
            energy=[]
            avgval=[]
            varience=[]
            first_derivate=[]
            dominantfreq=[]
            freqrange=[]
            if cover:
                dominantfreq2=[]
                freqrange2=[]
            features=[]
        pa.append(axi)
        axi=[]
    return np.array(pa)