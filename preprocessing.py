'''
This file provide needed functions for reading and preprocessing of the data
'''
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from scipy.signal import butter, filtfilt
import glob
import numpy as np
import os


# Denoising
def denoise_data(df, col_names, n_clusters):
    df_denoised = df.copy()
    for col_name, k in zip(col_names, n_clusters):
        clf = KNeighborsRegressor(n_neighbors=k, weights='uniform')
        clf.fit(df_denoised.index.values[:, np.newaxis], df_denoised[col_name])
        y_pred = clf.predict(df_denoised.index.values[:, np.newaxis])
        df_denoised[col_name] = y_pred
        # ax = pd.Series(df[col_name]).plot(color='lightgray')
        # # pd.Series(df["events"]).plot(color='g', ax=ax)
        # pd.Series(y_pred).plot(color='black', ax=ax, figsize=(12, 8))
        # plt.title("Denoising data")
    return df_denoised

# Common Average Reference (CAR) 
def apply_car(df, col_names):
    df_car = df.copy()
    for idx, row in df.iterrows():
        mean_val = row[col_names].mean()
        df_car.loc[idx, col_names] = row[col_names] - mean_val
    return df_car

# Z_scoring
def z_score(df, col_names):
    df_standard = df.copy()
    for col in col_names:
        df_standard[col] = (df[col] - df[col].mean()) / df[col].std()

    return df_standard

# Min-Max scaling
def min_max_scale(df, columns):
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val)
    return df

# Detrending
def detrend_signal(df, col_names):
    df_detrended = df.copy()
    for col in col_names:
        y = df_detrended[col]
        x = np.arange(len(y))
        p = np.polyfit(x, y, 1)
        trend = np.polyval(p, x)
        detrended = y - trend
        df_detrended[col] = detrended

    return df_detrended

# Band-pass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def Extract_channel_FreqBands(df):
    channels = df.columns.to_list()
    fs = 250  # Sampling rate
    bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, 40)}
    df_new = pd.DataFrame()

    for channel in channels:

        # Filter signal in frequency band of interest
        for i in bands:
            f_low = bands[i][0];
            f_high = bands[i][1];
            x_filt = butter_bandpass_filter(df[channel], f_low, f_high, fs)
            df_new[channel + '_' + i] = x_filt

    return df_new

# Applying the preprocessing functions and concatinating the dataframes
def read_data_and_preprocessing(path, events=[0, 1, 2], low_f=0.5, high_f=40, denoise=0, detrend=True, FreqBands=False, minmax=False, zscore=True, baseline_correction=(None, None)):
    eeg_files = glob.glob(os.path.join(path, "eeg/*eeg.csv"))
    event_files = glob.glob(os.path.join(path, "event/*event.csv"))

    eeg_files = sorted(eeg_files)
    event_files = sorted(event_files)

    if FreqBands:
        df = pd.DataFrame(columns=['P3_t_delta', 'P3_t_theta', 'P3_t_alpha', 'P3_t_beta', 'P3_t_gamma',
                                   'C3_t_delta', 'C3_t_theta', 'C3_t_alpha', 'C3_t_beta', 'C3_t_gamma',
                                   'C4_t_delta', 'C4_t_theta', 'C4_t_alpha', 'C4_t_beta', 'C4_t_gamma',
                                   'F5_t_delta', 'F5_t_theta', 'F5_t_alpha', 'F5_t_beta', 'F5_t_gamma',
                                   'P3_e_delta', 'P3_e_theta', 'P3_e_alpha', 'P3_e_beta', 'P3_e_gamma',
                                   'C3_e_delta', 'C3_e_theta', 'C3_e_alpha', 'C3_e_beta', 'C3_e_gamma',
                                   'C4_e_delta', 'C4_e_theta', 'C4_e_alpha', 'C4_e_beta', 'C4_e_gamma',
                                   'F5_e_delta', 'F5_e_theta', 'F5_e_alpha', 'F5_e_beta', 'F5_e_gamma',
                                   'events'])

    else:
        df = pd.DataFrame(columns=['P3_t', 'C3_t', 'C4_t', 'F5_t', 'P3_e', 'C3_e', 'C4_e', 'F5_e', 'events'])

    # loop over the list of csv files
    for eeg, event in zip(eeg_files, event_files):

        # read the csv file
        eeg_df = pd.read_csv(eeg)
        event_df = pd.read_csv(event)

        # Remove spaces
        eeg_df.columns = eeg_df.columns.str.replace(' ', '')

        # Select useful columns
        eeg_df = eeg_df[["FZ", "FC1", "FC2", "C3", "CZ", "C4", "CPZ", "PZ"]]

        # Rename columns
        eeg_df.columns = ['P3_t', 'C3_t', 'C4_t', 'F5_t', 'P3_e', 'C3_e', 'C4_e', 'F5_e']

        # Bandpass filter on raw data
        for column in eeg_df.columns:
            eeg_df[column] = butter_bandpass_filter(eeg_df[column], lowcut=low_f, highcut=high_f, fs=250)

        # Baseline Correction
        if baseline_correction[0] and baseline_correction[1]:
            baseline_start, baseline_end = baseline_correction 
            for column in eeg_df.columns:
                baseline_mean = np.mean(eeg_df[column][baseline_start:baseline_end])
                eeg_df[column] = eeg_df[column] - baseline_mean


        # Extract FreqBands
        if FreqBands:
            eeg_df = Extract_channel_FreqBands(eeg_df)

        # CAR
        # eeg_df = apply_car(eeg_df, list(eeg_df.columns))

        # Z-score
        if zscore:
            eeg_df = z_score(eeg_df, list(eeg_df.columns))

        # Min-Max scaling
        if minmax:
            eeg_df = min_max_scale(eeg_df, list(eeg_df.columns))

        # Detrending
        if detrend:
            eeg_df = detrend_signal(eeg_df, list(eeg_df.columns))

        # Denoising
        if denoise:
            eeg_df = denoise_data(eeg_df, list(eeg_df.columns), n_clusters=[denoise]*len(eeg_df.columns))

        # Concat
        df_concat = pd.concat([eeg_df, event_df], axis=1)

        # Remove Nan rows
        df_concat = df_concat.dropna(axis=0)

        # Remove rows with glass_event == 0 (because actually we don't need them)
        df_concat = df_concat[df_concat.glass_event != 0].reset_index()

        # Select usefull columns
        df_concat = df_concat[list(df.columns)]

        # Remove unselected events
        for i in range(3):
            if i not in events:
                df_concat = df_concat[df_concat.events != i]

        df = df.append(df_concat)

    df = df.reset_index()
    df = df.drop('index', axis=1)

    return df