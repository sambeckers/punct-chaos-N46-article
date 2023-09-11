#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 07:49:46 2023

@author: sam
"""
from amuse.lab import *
from pylab import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.signal import butter, lfilter, freqz
import os

# Set plt font to LaTeX
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "helvetica"
})

os.chdir("/Users/sam/Library/Mobile Documents/com~apple~CloudDocs/Sterrenkunde BSc 2022-2023 Jaar 3/Bachelor Research Project/Punctuated_Chaos/data_processed_N46")

#Lowpass filter functions:
def butter_lowpass(cutoff, fs, order):
    """
    nyq = nyquist frequency
    Parameters
    ----------
    cutoff : cutoff frequency
    fs : sampling frequency.
    order : bandpass order.

    Returns
    -------
    b : coefficient to apply to lfilter.
    a : coefficient to apply to lfilter.

    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq #normalised cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order):
    """
    Links the low bandapss to the data for the filter on the data
    """
    b, a = butter_lowpass(cutoff, fs, order)
    y = lfilter(b, a, data)
    return y

def lyapunov_fit(x,y):
    """
    Calculate a 1st order polynomial fit to the data and calculate the lyapunov timescale

    Parameters
    ----------
    x : time array.
    y : dxv or sma array

    Returns
    -------
    f : exponential polynomial fit (y-axis)
    lyapunov_timescale : t_lambda [yr]
    sigma_t_lambda : standard deviation on t_lambda [yr].

    """
    coeffs, covariance_matrix = np.polyfit(x, y, 1, cov=True)
    lyapunov_timescale = 1 / (coeffs[0]) #yr
    
    # Get the standard deviation of each coefficient:
    coeff_stdevs = np.sqrt(np.diag(covariance_matrix))
    
    sigma_t_lambda = coeff_stdevs[0] / (coeffs[0]**2) #simga_t_lambda = sigma_a / a^2
    f = np.exp(coeffs[0]*x + coeffs[1])
    return f, lyapunov_timescale, sigma_t_lambda

def plot_phase_space_distance(dxv_data, title):
    '''
    Plot phase space distance as function of time 
    Apply low-bandpass filter
    Apply LSQ FIT TO low-bandpass filtered data and calculate Lyapunov timescale
    '''
    plt.figure(dpi=450)
    time, dx, dv, dxv = dxv_data
    plt.plot(time[1:], dxv[1:], lw=1, c='k', alpha=0.5)
    plt.xlabel(r'\rm{t [yr]}')
    plt.ylabel(r'$\delta$')
    #plt.title(title)
    plt.yscale('log')
    
    time, psd = np.array(time[1:]), np.array(dxv[1:])
    # Setting standard filter requirements.
    order = 1
    fs = 1000.0
    cutoff = 0.6
    psd_f = butter_lowpass_filter(psd, cutoff, fs, order)
    plt.plot(time, psd_f, lw=1, c='orange', label=r"\rm{low-bandpass filtered}") 
    
    #Fix divide by zero in log of psd_f: 
    idx = np.where(psd_f > 0)
    psd_f = psd_f[idx[0][0]:]
    time = time[idx[0][0]:]
    
    f, lyapunov_timescale, stl = lyapunov_fit(time, np.log(psd_f))
    
    plt.plot(time, f, ls=":", lw=3, c='red', label=r"$t_\lambda \simeq$ \rm{{{}}} yr".format(round(lyapunov_timescale,1)))
    plt.legend()
    plt.savefig("/Users/sam/Documents/GitHub/punct-chaos-N46-article/Figures/Phase-space distance as function of all S-stars")
    plt.show()


#Load the file for the orbital parameters of the S-stars as a function of time of the canconical solution
data_orbits_can = pd.read_pickle("Sstar_cluster_N46MfromKmag_t2021yr_i0003_Tol-20.0_orbits.pkl")
df_orbits_can = pd.DataFrame(data_orbits_can)
df_orbits_can.index = ['t [yr]', 'SMA [au]', 'e []', 'i [deg]', 'TA [deg]', 'LAN', 'AOP'] #change the name of the rows of the dataframe
#print(df_orbits_can)
#print(len(data_orbits_can[1][0]))

#Load the file for the orbital parameters of the S-stars as a function of time of 
#the solution in which star S5 is perturbed in the x-direction by $10^{-10}$.
data_orbits_per = pd.read_pickle("Spstar_cluster_N46MfromKmag_t2021yr_i0003_Tol-20.0_orbits.pkl")
df_orbits_per = pd.DataFrame(data_orbits_per)
df_orbits_per.index = ['t [yr]', 'SMA [au]', 'e []', 'i [deg]', 'TA [deg]', 'LAN', 'AOP']
#print(df_orbits_S5)

time_list, sma_can, sma_per = data_orbits_can[0], data_orbits_can[1].value_in(units.au), data_orbits_per[1].value_in(units.au)

def calculate_plot_delta_a(time_c, sma_c, sma_p):
    """
    Calculates delta_a for each S-star
    Calculates the average delta_a for all stars
    Plots delta_a as a function of time

    Parameters
    ----------
    time_c : time list from 0 to 100000 in time steps of 10^-24
    sma_c : semi-major axis list of canonical solution.
    sma_p : semi-major axis list of perturbed solution with S5 given dx=10^-10 AU
    """
    sma_c = np.array(sma_c)
    sma_p = np.array(sma_p)
    time = []

    delta_a_lists = [[] for i in range(45)] #create 45 empty lists inside a large list
    
    #Calculation for each individual star:
    for n1 in range(len(sma_c)): #(0,10001)
        for n2 in range(len(sma_c[n1])): #(0,27)
            delta_a = np.sqrt(np.abs((sma_c[n1, n2] - sma_p[n1, n2]))**2)
            delta_a_lists[n2].append(delta_a) #appends result to the corresponding list, s.t. each star gets its corresponding results
        time.append(time_c[n1].value_in(units.yr))

    #Average of all stars:
    da = []
    for n1 in range(len(sma_c)):
        n = len(sma_c[n1])
        dai = 0
        for n2 in range(len(sma_c[n1])):
            dai += (sma_c[n1][n2]-sma_p[n1][n2])**2
        da.append(np.sqrt(dai)/(n))
        if da[-1]==0:
            da[-1] = 1.e-25
    
    time = np.array(time)
    f_total, lyapunov_timescale_total, stl_total = lyapunov_fit(time, np.log(da))
    
    # idx_after = np.where(time > 2876)
    # f_after, lyapunov_timescale_after, stl_after = lyapunov_fit(time[idx_after[0][0]:], np.log(da[idx_after[0][0]:]))
    
    # idx_before = np.where(time < 2876)
    # f_before, lyapunov_timescale_before, stl_before = lyapunov_fit(time[:idx_before[0][-1]], np.log(da[:idx_before[0][-1]]))
    
    plt.figure(dpi=450)
    for i in range(len(delta_a_lists)):
        plt.plot(time, np.log(delta_a_lists[i]), lw=1, alpha=0.5)
    plt.plot(time, np.log(da), c='k', lw=1, label='$\overline{\delta_a}$')
    plt.plot(time, np.log(f_total), linestyle="dashdot", lw=2, c='red', label=r"$t_\lambda \simeq$ \rm{{{}}} yr".format(round(lyapunov_timescale_total,1)))
    #plt.plot(time[idx_after[0][0]:], np.log(f_after), linestyle="dashdot", lw=2, c='orange', label=r"$t_\lambda \simeq$ \rm{{{}}} yr".format(round(lyapunov_timescale_after,1)))
    #plt.plot(time[:idx_before[0][-1]], np.log(f_before), linestyle="dashdot", lw=2, c='mediumblue', label=r"$t_\lambda \simeq$ \rm{{{}}} yr".format(round(lyapunov_timescale_before,1)))
    plt.xlabel(r'\rm{Time [yr]}')
    plt.ylabel(r'$\log{\delta_a \rm{[au]}}$')
    plt.legend(loc='lower right')
    #plt.yscale('log')
    #plt.title('Time evolution of separation in semi-major axis\nfor each individual S-star,\nwith the average separation of all stars indicated')
    plt.savefig("/Users/sam/Documents/GitHub/punct-chaos-N46-article/Figures/separation in semi-major axis for each S-star + avg separation.pdf")
    plt.show()

calculate_plot_delta_a(time_list, sma_can, sma_per)

# Read the data
sstars = pd.read_pickle("BH_sstars_t_dr_dv_drv_inclN_body_i0003.pkl")
df_sstars = pd.DataFrame(sstars)
df_sstars.index = ['t [yr]', 'dr','dr_Nbody', 'dv', 'dv_Nbody', 'drv_Nbody']
#print(df_sstars)

S_name = ['BH','S1', 'S2', 'S4', 'S6', 'S8', 'S9', 'S12', 'S13', 'S14', 'S17', 'S18',
          'S19', 'S21', 'S22', 'S23', 'S24', 'S29', 'S31', 'S33', 'S38', 'S39',
          'S42', 'S54', 'S55', 'S60', 'S62', 'S66', 'S67', 'S71', 'S83', 'S85',
          'S87', 'S89', 'S91', 'S96', 'S97', 'S145', 'S175', 'S4711', 'S4712',
          'S4713', 'S4714', 'S4715', 'R34', 'R44']
colors = ['k', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
          '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
          '#c49c94', '#f7b6d2', '#85144b', '#dbdb8d', '#9edae5',
          '#393b79', '#fe7f0e', '#6b6ecf', '#f0027f', '#7f3b08',
          '#01FF70', '#5F9EA0', '#FFD700', '#FF00FF', '#00FFFF',
          '#008000', '#800080', '#000080', '#FFFF00', '#FF6600',
          '#66FF66', '#660066', '#006600', '#666600', '#006666',
          '#9966CC', '#FF5050', '#003300', '#3366CC', '#339966',
          '#CC0033', '#FF9966']

plt.figure(dpi=450)
for i, color in zip(range(0,46), colors):
    plt.plot(sstars[0][1:], np.log10(sstars[1][i][1:]), lw=1, c=color)
plt.xlabel(r'\rm{Time [yr]}')
plt.ylabel(r'$\log_{10}(\delta_r)$ \rm{[au]}')
#plt.title(r'\rm{Separation in position space}')
"""
Change the legend parameters s.t. it is next to the plot and has two columns. 
Loop through the colors and names to assign them to patches
"""
plt.legend(handles = [Patch(facecolor=color, edgecolor='black', label=r'\rm{{{}}}'.format(name)) for color, name in zip(colors, S_name)], loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
plt.grid()
plt.savefig("/Users/sam/Documents/GitHub/punct-chaos-N46-article/Figures/separation_in_position_space with labels.pdf", bbox_inches="tight")
plt.show()