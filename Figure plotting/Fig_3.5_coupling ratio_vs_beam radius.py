# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 12:34:03 2020

@author: liuqi
"""
import scipy as sp
import numpy as np

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pylab as plt
import scipy.integrate as spi
from scipy.integrate import quad
import seaborn
from scipy.integrate import odeint
from scipy.integrate import dblquad
import Will_Module_addwdep as TQ
import Module_table_parameter as MTP
import Module_integration_manually as MIM

import Module_Fz_stiffness_at_parameter as MFSAP

import time




integration_method = 'manual'   # 'manual' or 'integrated'
grid_size = 200

plt.close('all')

###########################

#Our sphere

g = 9.8

c = 3 * 10**8

w_0 = 2 * 10 ** (-6)
#w_0 = 0.85 * 10 ** (-6)  #optimal w0 for stiffness matching

Lambda = 1.064 * 10**(-6)

z_R = np.pi* w_0 ** 2 / Lambda

rho = 30 * 10 ** (-6)
 
weight = 3.4 * 10**(-10)

n_0 = 1

n_s_n = 0.04

k = 7.6097

n_s = n_s_n - k*1j

sig_s = 10.49 * 10 ** 3 * (( 3 ** 3 - 2.25 ** 3) / 3 ** 3 ) #density of sphere in kg/m^3

sig_0 = 0 #density of medium in kg/m^3

m = 4/3 * np.pi * rho ** 3 * ( sig_s - sig_0 )

Permittivity = 8.85 * 10**(-12)

#P = 0.5 * c * n_0 * Permittivity    #total power of the LG01 beam

P = 12.03    #optimal power for stiffness matching and stable equilibrium

################################################################################################################
#FIG 3.5 Investigation of axial force change vs axial offsets(coupling ratio vs beam radius w at target bottom )
################################################################################################################
#x-axis: beam radius at target bottom
#y-axis: coupling radio

#key function: TQ.Fz_total_vs_rho0x_plot, TQ.F_total_manual_integration


rho_0 = [0,0]   #no offset

w = np.linspace(0.4* np.sqrt(2)*rho, 2*np.sqrt(2)*rho, 50) #target bottom beam radius

rho_00 = np.linspace(-6*rho, 6*rho, 300)   #x axis radial offset

ratio = []

axialmin = []
axialmax = []
diffe = []
w_minloc = []

for we in w:
    
    axial_force_array = np.asarray(TQ.Fz_total_vs_rho0x_plot(rho_00,rho_0[1], rho, n_0, n_s, w_0, we, z_R, P, target = "reflective", integration_method = integration_method, grid_size = grid_size))
    
    Q_zmin = axial_force_array  * c / ( n_0 * P ) #Qz array

    axial_min = np.min(Q_zmin)  #min of Qz array
    
    axialmin.append(axial_min)
    
    min_location = (np.sqrt(2)*rho_00 / we)[np.argmin(Q_zmin)]
    
    w_minloc.append(min_location)
    
    axial_centre_max = TQ.F_total_manual_integration(0, rho_0[1], rho, n_0, n_s, w_0, we, z_R, P, target = "reflective", coordinate = 'z', grid_size = grid_size)['force_total']   #Fz at no offset
    
    Q_zmax = axial_centre_max  * c / ( n_0 * P )  #trapping efficiency at no offset
    
    axialmax.append(Q_zmax)
    
    diff = Q_zmax - axial_min
    
    diffe.append(diff)
    
    fluct_ratio = (Q_zmax - axial_min)/abs(Q_zmax)   #REFERENCE Eq.(3.2) coupling ratio
    
    ratio.append(fluct_ratio)

print(axialmin)
print(axialmax)
print(diffe)
print(w_minloc)


plt.plot(w/(np.sqrt(2)*rho), ratio , lw=2, c="c", label=r'$w = w_{0}$')

plt.title(r"$\rho = 30um, w_{0} = 2um$",fontsize=20)

plt.xlabel('w/(sqrt(2)rho)',fontsize=20)
plt.ylabel('ratio',fontsize=20)
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
plt.grid()
plt.show()