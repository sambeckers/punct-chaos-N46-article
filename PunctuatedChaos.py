#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 07:49:46 2023

@author: sam
"""

import pandas as pd

data_orbits_can = pd.read_pickle("Spstar_cluster_N46MfromKmag_t2021yr_i0000_Tol-20.0_orbits.pkl")
df_orbits_can = pd.DataFrame(data_orbits_can)
df_orbits_can.index = ['t [yr]', 'SMA [au]', 'e []', 'i [deg]', 'TA [deg]', 'LAN', 'AOP'] #change the name of the rows of the dataframe
print(df_orbits_can)
print(len(data_orbits_can[1][0]))
