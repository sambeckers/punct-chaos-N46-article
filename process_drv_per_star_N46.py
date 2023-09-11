#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 16:52:17 2023

@author: sam
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 11:38:27 2023

@author: sam
"""
from amuse.lab import *
from matplotlib import pyplot
import pickle
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

start_time = time.time() # Start time on running time

def get_particle_data(canonical, perturbed, outfilename):
    """
    Calculates dr, dv and drv for the black hole + 27 S-stars for all timesteps up to 10000 yr. 
    Converts dr and dv to N-body units s.t. drv can be calculated in N-body units
    Saves list with time, dr, dr_Nbody, dv, drv_Nbody to a .pkl file

    Parameters
    ----------
    canonical : dataset with the canonical solutions of dynamical evolution of the S-stars in the galactic centre.
    perturbed : dataset with perturbed solutions of dynamical evolution of the S-stars in the galactic centre, 
    but with star S5 perturbed by dx = 15 m .
    outfilename : the file which the data is going to be written to.

    Returns
    -------
    star_df : dataframe of the calculated data.
    star_array : list with time, dr, dr_Nbody, dv, drv_Nbody.
    """
    index = 0 # Index for keeping up with timestep
    
    # Initalise lists:
    time = [] #[yr]
    star_dr = [[] for i in range(46)] #[au] #An empty list with 28 empty nestled lists inside
    star_dv = [[] for i in range(46)] #[km/s]
    star_dr_Nbody = [[] for i in range(46)] #[N-body length] #List for values in N_body units
    star_dv_Nbody = [[] for i in range(46)] #[N-body (length/time)]
    star_drv_Nbody = [[] for i in range(46)] #[N-body sqrt(length**2 + (length/time)**2)] 
    
    # Setting conditions for the unitconverter. We need this in order to get drv:
    """
    In order to do so use N-body units, in that case r and v are scaled
    s.t. one variable does not dominate the other, i.e. r is in AU, a much
    larger quantity than velocity (in size)
    """
    total_mass = sum(canonical.mass.value_in(units.MSun)) | units.MSun # sum of all masses in system (first get value without units with value.in()) and then apply units.MSun on sum()
    #print(total_mass)
    scale_length = 0.01 | units.pc # "the S-star cluster of B-type stars at a galactocentric distance of ∼0.01 pc" https://arxiv.org/abs/2002.10547
    """
    nbody_system.nbody_to_si(), the converter, takes the total of the system M and scale length as arguments.
    The scale length is a length that allows the system to calculate the conversions in a certain order or magnitude
    As long as this value is within a few orders of magnitude of the true lengths in the system, the conversion will be correct
    """
    converter = nbody_system.nbody_to_si(total_mass, scale_length)
    
    """
    .history() retrieves all time 'snapshots' of the data. With the for loop we iterate through each timestep
    tqdm() keeps track of total iterations performed and iterations/second
    """
    for ball_c, ball_p in tqdm(zip(canonical.history, perturbed.history)):
        index += 1 # goes up by 1 for every timestep (so up to 100002)
        time.append(0.1*index)
        
        for obj in range(46): # There is 1 black hole + 27 stars
            p_c = Particles() # Initialise Particle object for canonical (c) solution
            p_p = Particles() # Initialise Particle object for perturbed (p) solution
            # Add one particle of the iteration. Take the index as each ball_c is a timestep
            # in which each particle is located (e.g. if obj = 1, it will get the data for S1 at the time step the above for loop is at):
            p_c.add_particle(ball_c[obj])
            p_p.add_particle(ball_p[obj]) 
        
            # Calculate dr by determining the length of the vector
            dr = np.sqrt((p_c[0].x-p_p[0].x)**2 + (p_c[0].y-p_p[0].y)**2 + (p_c[0].z-p_p[0].z)**2) # [0] gives only the value without [] around it
            star_dr[obj].append(dr.value_in(units.au)) # append the value only to the nestled list for the black hole/star for which dr has been calculated
            
            # Convert to N-body units and take the value only (in N-body length unit)
            dr_Nbody = converter.to_nbody(dr).value_in(nbody_system.length) 
            star_dr_Nbody[obj].append(dr_Nbody)
            
            # Now calculate for dv in similar fashion to dr
            dv = np.sqrt((p_c[0].vx-p_p[0].vx)**2 + (p_c[0].vy-p_p[0].vy)**2 + (p_c[0].vz-p_p[0].vz)**2)
            star_dv[obj].append(dv.value_in(units.kms))
            dv_Nbody = converter.to_nbody(dv).value_in(nbody_system.speed)
            star_dv_Nbody[obj].append(dv_Nbody)
            
            # Calculate drv in N-body units by taking dr and dv in N-body units
            drv_Nbody = np.sqrt((dr_Nbody)**2 + (dv_Nbody)**2) / 2
            star_drv_Nbody[obj].append(drv_Nbody)
            
            # Remove particle from Particles() (in every iteration) as we only want 1 particle in it per iteration
            p_c.remove_particle(ball_c[obj])
            p_p.remove_particle(ball_p[obj])
    
    star_array = [time, star_dr, star_dr_Nbody, star_dv,star_dv_Nbody, star_drv_Nbody]
    star_df = pd.DataFrame(star_array)
    #print(star_df)
    
    # Save the data to a file
    file = open(outfilename, "wb")
    pickle.dump(star_array, file)
    file.close()
    return star_df, star_array

# Initiating the script

def new_option_parser():
    """
    Command-line parser. Used for parsing the data files s.t. they can be used in 'main' and parsing the 
    output file s.t. the data can be written to it

    Returns
    -------
    result : contains the information for all data files, which can be accessed with new_option_parser().parse_args().

    """
    from amuse.units.optparse import OptionParser
    result = OptionParser()
    result.add_option("--f_can", dest="filename_canonical",
                      default = "../data_raw_N46/Sstar_cluster_N46MfromKmag_t2021yr_i0003_Tol-20.0.amuse",
                      help="input outfilename  [%default]")
    result.add_option("--f_pert", dest="filename_perturbed",
                      default = "../data_raw_N46/Spstar_cluster_N46MfromKmag_t2021yr_i0003_Tol-20.0.amuse",
                      help="input outfilename  [%default]")
    result.add_option("-F", dest="outfilename",
                      default = "../data_processed_N46/BH_sstars_t_dr_dv_drv_inclN_body_i0003.pkl",
                      help="input outfilename  [%default]")
    return result


if __name__ in ('__main__'):
    o, arguments  = new_option_parser().parse_args()
    canonical = read_set_from_file(o.filename_canonical, "amuse", close_file=True) # read the file of the canonical solution
    perturbed = read_set_from_file(o.filename_perturbed, "amuse", close_file=True) # read the file of the perturbed solution
    star_df, star_array = get_particle_data(canonical, perturbed,  o.outfilename)
    #print(len(star_array[1]))
    #print(max(star_array[1]))

print("--- %s minutes ---" % ((time. time() - start_time) / 60))