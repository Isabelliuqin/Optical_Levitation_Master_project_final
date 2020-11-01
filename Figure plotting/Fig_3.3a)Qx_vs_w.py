# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 18:50:25 2020

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



integration_method = 'manual'   # 'manual' or 'integrated'
grid_size = 100

plt.close('all')

###########################

#Our sphere

g = 9.8 

c = 3 * 10**8

w_0 = 2 * 10 ** (-6)

Lambda = 1.064 * 10**(-6)

z_R = np.pi* w_0 ** 2 / Lambda

rho = 30 * 10 ** (-6)
 
n_0 = 1

n_s_n = 0.04

k = 7.6097

n_s = n_s_n - k*1j

sig_s = 10.49 * 10 ** 3 * (( 3 ** 3 - 2.25 ** 3) / 3 ** 3 ) #density of sphere in kg/m^3

sig_0 = 0 #density of medium in kg/m^3

m = 4/3 * np.pi * rho ** 3 * ( sig_s - sig_0 )

Permittivity = 8.85 * 10**(-12)

#P = 0.5 * c * n_0 * Permittivity    #total power of the LG01 beam

P = 12  #optimal power required to levitate at w0 = 0.85um


#####################################################
#FIG 3.3a) & FIG 3.6  plot Qx vs w for various rho_0x
#####################################################

#x-axis: beam radius w at target bottom
#y-axis Qx trapping efficiency

#key function: TQ.Fx_total_vs_d_plot


w = np.linspace(0.5*np.sqrt(2)*rho, 2*np.sqrt(2)*rho, 100) #beam radius

rho_0 = [0, 0]

rho_0[0] = [0.05*rho, 0.25*rho, 0.5*rho, 0.75*rho, rho, 1.25*rho, 2*rho]  #various x-axis offset





radial_flistx0 = np.asarray(TQ.Fx_total_vs_d_plot(rho_0[0][0],rho_0[1], rho, n_0, n_s, w_0, w, z_R, P, target = "reflective", integration_method = integration_method, grid_size = grid_size))

Q_x0 = radial_flistx0  * c / ( n_0 * P )


radial_flistx1 = np.asarray(TQ.Fx_total_vs_d_plot(rho_0[0][1],rho_0[1], rho, n_0, n_s, w_0, w, z_R, P, target = "reflective", integration_method = integration_method, grid_size = grid_size))

Q_x1 = radial_flistx1  * c / ( n_0 * P )


radial_flistx2 = np.asarray(TQ.Fx_total_vs_d_plot(rho_0[0][2],rho_0[1], rho, n_0, n_s, w_0, w, z_R, P, target = "reflective", integration_method = integration_method, grid_size = grid_size))

Q_x2 = radial_flistx2  * c / ( n_0 * P )

radial_flistx3 = np.asarray(TQ.Fx_total_vs_d_plot(rho_0[0][3],rho_0[1], rho, n_0, n_s, w_0, w, z_R, P, target = "reflective", integration_method = integration_method, grid_size = grid_size))

Q_x3 = radial_flistx3  * c / ( n_0 * P )

radial_flistx4 = np.asarray(TQ.Fx_total_vs_d_plot(rho_0[0][4],rho_0[1], rho, n_0, n_s, w_0, w, z_R, P, target = "reflective", integration_method = integration_method, grid_size = grid_size))

Q_x4 = radial_flistx4  * c / ( n_0 * P )


radial_flistx5 = np.asarray(TQ.Fx_total_vs_d_plot(rho_0[0][5],rho_0[1], rho, n_0, n_s, w_0, w, z_R, P, target = "reflective", integration_method = integration_method, grid_size = grid_size))

Q_x5 = radial_flistx5  * c / ( n_0 * P )

radial_flistx6 = np.asarray(TQ.Fx_total_vs_d_plot(rho_0[0][6],rho_0[1], rho, n_0, n_s, w_0, w, z_R, P, target = "reflective", integration_method = integration_method, grid_size = grid_size))

Q_x6 = radial_flistx6  * c / ( n_0 * P )


plt.figure(9)
#plt.plot( w / (np.sqrt(2) * rho), Q_x0, lw=2, c="c", label="rho_0x/rho = 0.05")

#plt.plot( w  / (np.sqrt(2) * rho), Q_x1, lw=2, c="r", label="rho_0x/rho = 0.25")

plt.plot( w  / (np.sqrt(2) * rho), Q_x2, lw=2, c="g", label="rho_0x/rho = 0.5")

plt.plot( w  / (np.sqrt(2) * rho), Q_x3, lw=2, c="y", label="rho_0x/rho = 0.75")

plt.plot( w  / (np.sqrt(2) * rho), Q_x4, lw=2, c="b", label="rho_0x/rho = 1")

plt.plot( w  / (np.sqrt(2) * rho), Q_x5, lw=2, c="r", label="rho_0x/rho = 1.25")

#plt.plot( w  / (np.sqrt(2) * rho), Q_x6, lw=2, c="m", label="rho_0x/rho = 2")



new_ticks1 = np.linspace(0.5, 2, 4) # plot axis
print(new_ticks1)
plt.xticks(new_ticks1,fontsize=20)
plt.yticks(np.linspace(-0.4, 0.2, 4),fontsize=20)

plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0.5))
ax.spines['bottom'].set_position(('data',0))

plt.legend(loc=4,fontsize=16)

plt.xlabel('w/(sqrt(2)rho)',fontsize=20)
plt.ylabel('Qx',fontsize=20)
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)

plt.title('rho = 30um, w0 = 2um',fontsize=20)
plt.grid()
plt.show()

