# Stimulating-States-of-Parkinsonian-Tremor

### Basic Summary:
For running the code, add your directory in main.py
  Incase you're using 3 axis or 1 axis, change the function name from ThreeAxisAcc to OneAxisAcc (both defined in the preprocessing_helper.py)
  Incase you want to change the preprocessing, consult  ACC_preprocessing_helper, sliding_ACC_preprocessing_helper, sliding_EMG_preprocessing_helper and EMG_preprocessing_helper     (all defined in preprocessing_helper.py). Also, you can change lowpass highpass filters etc.
  For performing the feature extraction on one patient all axis or all patients all axis, use the funtions 

For getting the preprocessed data and its matlab code, contact the authors
mmughees.bscs16seecs@seecs.edu.pk
timothy.west@ndcn.ox.ac.uk

Other details of the work are present in the presentation.

The .py files are to be used. If want to check the testing then refer to the Python Notebook files. For working in deeplearning there is an architecture called TimeNet. Working implementation is in the respective folder but its capabilities are yet to be searched.

![image](https://user-images.githubusercontent.com/32590595/135745124-0dcb8ba1-7189-4109-95ca-668fde4d748b.png)
