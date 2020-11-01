# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 17:03:09 2020

@author: liuqi
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 10:41:09 2020

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

w_0 = 2 * 10 ** (-6)
#w_0 = 0.85 * 10 ** (-6)  #optimal w0 for stiffness matching

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


#####################################################################################
#FIG 3.2a)   plot of Q_z vs beam radius at target bottom w for various offset rho_0x
#####################################################################################

#x-axis: beam radius at target bottom(normalised)
#y-axis: Qz trapping efficiency
#plotted at various radial offsets

#key functions: TQ.Fz_total_vs_d_plot NOTE it is actually computing Fz at beam radius elements

rho_0 = [0,0]   


rho_0[0] = [0, 0.25*rho, 0.5*rho, 0.75*rho, rho, 1.25*rho, 2*rho]   #various x-axis radial offsets
w = np.linspace(w_0, 2*np.sqrt(2) * rho, 100)                       #beam radius at target bottom -- iterating



Axial_flist_vs_d0 =  np.asarray(TQ.Fz_total_vs_d_plot(rho_0[0][0],rho_0[1], rho, n_0, n_s, w_0, w, z_R, P, target = "reflective", integration_method = integration_method, grid_size = grid_size))

Q_z0 = Axial_flist_vs_d0 * c / ( n_0 * P )

Axial_flist_vs_d1 =  np.asarray(TQ.Fz_total_vs_d_plot(rho_0[0][1],rho_0[1], rho, n_0, n_s, w_0, w, z_R, P, target = "reflective", integration_method = integration_method, grid_size = grid_size))

Q_z1 = Axial_flist_vs_d1 * c / ( n_0 * P )

Axial_flist_vs_d2 =  np.asarray(TQ.Fz_total_vs_d_plot(rho_0[0][2],rho_0[1], rho, n_0, n_s, w_0, w, z_R, P, target = "reflective", integration_method = integration_method, grid_size = grid_size))

Q_z2 = Axial_flist_vs_d2 * c / ( n_0 * P )

Axial_flist_vs_d3 =  np.asarray(TQ.Fz_total_vs_d_plot(rho_0[0][3],rho_0[1], rho, n_0, n_s, w_0, w, z_R, P, target = "reflective", integration_method = integration_method, grid_size = grid_size))

Q_z3 = Axial_flist_vs_d3 * c / ( n_0 * P )

Axial_flist_vs_d4 =  np.asarray(TQ.Fz_total_vs_d_plot(rho_0[0][4],rho_0[1], rho, n_0, n_s, w_0, w, z_R, P, target = "reflective", integration_method = integration_method, grid_size = grid_size))

Q_z4 = Axial_flist_vs_d4 * c / ( n_0 * P )

Axial_flist_vs_d5 =  np.asarray(TQ.Fz_total_vs_d_plot(rho_0[0][5],rho_0[1], rho, n_0, n_s, w_0, w, z_R, P, target = "reflective", integration_method = integration_method, grid_size = grid_size))

Q_z5 = Axial_flist_vs_d5 * c / ( n_0 * P )

Axial_flist_vs_d6 =  np.asarray(TQ.Fz_total_vs_d_plot(rho_0[0][6],rho_0[1], rho, n_0, n_s, w_0, w, z_R, P, target = "reflective", integration_method = integration_method, grid_size = grid_size))

Q_z6 = Axial_flist_vs_d6 * c / ( n_0 * P )

plt.figure(1)




plt.plot( w / (np.sqrt(2)*rho), Q_z0, lw=2, c="r", label="rho_0x/rho = 0")
#plt.plot( w / (np.sqrt(2)*rho), Q_z1 , lw=2, c="r", label="rho_0x/rho = 0.25")
plt.plot( w / (np.sqrt(2)*rho), Q_z2 , lw=2, c="g", label="rho_0x/rho = 0.5")
plt.plot( w / (np.sqrt(2)*rho), Q_z3 , lw=2, c="y", label="rho_0x/rho = 0.75")
plt.plot( w / (np.sqrt(2)*rho), Q_z4, lw=2, c="b", label="rho_0x/rho = 1")
plt.plot( w / (np.sqrt(2)*rho), Q_z5 , lw=2, c="m", label="rho_0x/rho = 1.25")
#plt.plot( w / (np.sqrt(2)*rho), Q_z6 , lw=2, c="m", label="rho_0x/rho = 2")


new_ticks1 = np.linspace(0, 2, 5) # plot axis
print(new_ticks1)
plt.xticks(new_ticks1,fontsize=20)
plt.yticks(np.linspace(-2.0, 0.5, 6),fontsize=20)
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
plt.xlabel('(w/sqrt(2))/rho',fontsize=20)
plt.ylabel('Qz',fontsize=20)
plt.grid()
plt.show()


