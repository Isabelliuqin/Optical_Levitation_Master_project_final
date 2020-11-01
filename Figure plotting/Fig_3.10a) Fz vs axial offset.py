# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 00:38:04 2020

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


import time

############################
# integration method control
############################

integration_method = 'manual'   # 'manual' or 'integrated'
grid_size = 200

plt.close('all')

###########################

#Our sphere

g = 9.8

c = 3 * 10**8


w_0 = 0.85 * 10 ** (-6)  #optimal w0 for stiffness matching

Lambda = 1.064 * 10**(-6)

z_R = np.pi* w_0 ** 2 / Lambda

rho = 30 * 10 ** (-6)

n_0 = 1

n_s_n = 0.04

k = 7.6097

n_s = n_s_n - k*1j

sig_s = 10.49 * 10 ** 3 * (( 3 ** 3 - 2.25 ** 3) / 3 ** 3 ) #density of sphere in kg/m^3

sig_0 = 0                                                   #density of medium in kg/m^3

m = 4/3 * np.pi * rho ** 3 * ( sig_s - sig_0 )              #mass of sphere

Permittivity = 8.85 * 10**(-12)

#P_norm = 0.5 * c * n_0 * Permittivity  

P = 12    #optimal power for stiffness matching and stable equilibrium
################################################################################################
#FIG 3.10a)   plot of Q_z+Q_weight vs axial offset rho_0z from beam waist to target bottom
################################################################################################

#x-axis: axial offset rho_0z
#y-axis: Qz + Q_weight trapping efficiency + effect of target weight

#key function: TQ.Fz_total_vs_d_plot


rho_0 = [0,0]   #no offset


w = np.linspace(w_0, 60 * 10** (-6), 100)           #beam radius(iterating)

rho_0z = np.sqrt( (z_R ** 2 / w_0 ** 2) * ( w**2 - w_0**2 ))  #axial offset


Axial_flist_vs_d =  np.asarray(TQ.Fz_total_vs_d_plot(rho_0[0],rho_0[1], rho, n_0, n_s, w_0, w, z_R, P, target = "reflective", integration_method = integration_method, grid_size = grid_size))



Q_z = Axial_flist_vs_d * c / ( n_0 * P )
Q_weight = m*g*c/(n_0 * P)

plt.figure(3)


plt.plot( rho_0z * 10 ** (6), Q_z + Q_weight , lw=2, c="c", label="rho_0x = 0")


new_ticks1 = np.linspace(0, 160, 9) # plot axis
print(new_ticks1)
plt.xticks(new_ticks1,fontsize=20)
plt.yticks(np.linspace(-2, 0.5, 6),fontsize=20)

plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
ax.spines['bottom'].set_position(('data',0))

plt.legend(loc=4,fontsize=16)


plt.xlabel('rho_0z(um)',fontsize=20)
plt.ylabel('Qz + Qw',fontsize=20)
plt.title('rho = 30um, w0 = 0.85um',fontsize=20)
plt.grid()
plt.show()