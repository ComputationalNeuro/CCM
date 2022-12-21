import numpy as np
from scipy.io import loadmat  # SciPy module that loads mat-files
import matplotlib.pyplot as plt
from datetime import datetime, date, time
import pandas as pd

from pyEDM import *
from causal_ccm import *
from tqdm import tqdm # for showing progress bar in for loops

#get single trials spike train, index = trial #
def get_single_trial_data(spikes, index):
    trial_spikes = []
    for train in spikes:
        trial_spikes.append(train[index])
    return trial_spikes


# only include time bins 101 through 181
def filter_time_bins(spikes):
    relevant_spikes = []
    for train in spikes:
        spike_train = []

        for trial in train:
            spike_train.extend(trial[101:181].tolist())

        relevant_spikes.append(spike_train)
    return relevant_spikes

# Function for Applying Moving Average Time Series Smoother
def moving_average(spikes, kernel_size):
    smoothed = np.empty([585, 2400])
    i = 0
    for train in spikes:
        kernel = np.ones(kernel_size) / kernel_size
        smoothed[i] = np.convolve(train, kernel, mode='same')
        i+=1
    return smoothed
    
def get_neuron_groups(neurons):

    aNeurons = []
    bNeurons = []
    cNeurons = []
    dNeurons = []
    eNeurons = []
    fNeurons = []
    gNeurons = []
    hNeurons = []

    for n in range(0,584):
        if neurons[n][1][0] == 'A':
            aNeurons.append(n)
        if neurons[n][1][0] == 'B':
            bNeurons.append(n)
        if neurons[n][1][0] == 'C':
            cNeurons.append(n)
        if neurons[n][1][0] == 'D':
            dNeurons.append(n)
        if neurons[n][1][0] == 'E':
            eNeurons.append(n)
        if neurons[n][1][0] == 'F':
            fNeurons.append(n)
        if neurons[n][1][0] == 'G':
            gNeurons.append(n)
        if neurons[n][1][0] == 'H':
            hNeurons.append(n)

    return np.vstack((aNeurons, bNeurons, cNeurons, dNeurons, eNeurons, fNeurons, gNeurons, hNeurons))