# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 17:16:40 2020

@author: liuqi
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 10:58:43 2020

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


###############################################
#FIG 3.2b)  plot of Q_z vs rho_0x for various w
###############################################

#x-axis: rho_0x x-axis radial offset
#y-axis: Qz trapping efficiency

#key function: TQ.Fz_total_vs_rho0x_plot

rho_0 = [0,0]   #no offset


w = [w_0, 0.25* np.sqrt(2)*rho, 0.5 * np.sqrt(2)*rho, 0.75* np.sqrt(2)*rho,np.sqrt(2)*rho, 1.25*np.sqrt(2)*rho, 2*np.sqrt(2)*rho] #various beam radius

rho_00 = np.linspace(-3*w[0]/np.sqrt(2), 3*w[0]/np.sqrt(2), 100) #radial offset(iterating)

rho_01 = np.linspace(-3*w[1]/np.sqrt(2), 3*w[1]/np.sqrt(2), 100)

rho_02 = np.linspace(-3*w[2]/np.sqrt(2), 3*w[2]/np.sqrt(2), 100)

rho_03 = np.linspace(-3*w[3]/np.sqrt(2), 3*w[3]/np.sqrt(2), 100)

rho_04 = np.linspace(-3*w[4]/np.sqrt(2), 3*w[4]/np.sqrt(2), 100)

rho_05 = np.linspace(-3*w[5]/np.sqrt(2), 3*w[5]/np.sqrt(2), 100)

rho_06 = np.linspace(-3*w[6]/np.sqrt(2), 3*w[6]/np.sqrt(2), 100)


Axial_flist_vs_d0 =  np.asarray(TQ.Fz_total_vs_rho0x_plot(rho_00,rho_0[1], rho, n_0, n_s, w_0, w[0], z_R, P, target = "reflective", integration_method = integration_method, grid_size = grid_size))

Q_z0 = Axial_flist_vs_d0 * c / ( n_0 * P )

Axial_flist_vs_d1 =  np.asarray(TQ.Fz_total_vs_rho0x_plot(rho_01,rho_0[1], rho, n_0, n_s, w_0, w[1], z_R, P, target = "reflective", integration_method = integration_method, grid_size = grid_size))

Q_z1 = Axial_flist_vs_d1 * c / ( n_0 * P )

Axial_flist_vs_d2 =  np.asarray(TQ.Fz_total_vs_rho0x_plot(rho_02,rho_0[1], rho, n_0, n_s, w_0, w[2], z_R, P, target = "reflective", integration_method = integration_method, grid_size = grid_size))

Q_z2 = Axial_flist_vs_d2 * c / ( n_0 * P )

Axial_flist_vs_d3 =  np.asarray(TQ.Fz_total_vs_rho0x_plot(rho_03,rho_0[1], rho, n_0, n_s, w_0, w[3], z_R, P, target = "reflective", integration_method = integration_method, grid_size = grid_size))

Q_z3 = Axial_flist_vs_d3 * c / ( n_0 * P )

Axial_flist_vs_d4 =  np.asarray(TQ.Fz_total_vs_rho0x_plot(rho_04,rho_0[1], rho, n_0, n_s, w_0, w[4], z_R, P, target = "reflective", integration_method = integration_method, grid_size = grid_size))

Q_z4 = Axial_flist_vs_d4 * c / ( n_0 * P )

Axial_flist_vs_d5 =  np.asarray(TQ.Fz_total_vs_rho0x_plot(rho_05,rho_0[1], rho, n_0, n_s, w_0, w[5], z_R, P, target = "reflective", integration_method = integration_method, grid_size = grid_size))

Q_z5 = Axial_flist_vs_d5 * c / ( n_0 * P )

Axial_flist_vs_d6 =  np.asarray(TQ.Fz_total_vs_rho0x_plot(rho_06,rho_0[1], rho, n_0, n_s, w_0, w[6], z_R, P, target = "reflective", integration_method = integration_method, grid_size = grid_size))

Q_z6 = Axial_flist_vs_d6 * c / ( n_0 * P )



plt.figure(1)


#plt.plot(np.sqrt(2)*rho_00 / w[0], Q_z0 , lw=2, c="c", label='w = w0')
#plt.plot(np.sqrt(2)*rho_01 / w[1], Q_z1 , lw=2, c="r", label='(w/sqrt{2})/rho = 0.25')
plt.plot(np.sqrt(2)*rho_02 / w[2], Q_z2 , lw=2, c="g", label="w/(sqrt(2)rho) = 0.5")
plt.plot(np.sqrt(2)*rho_03 / w[3], Q_z3 , lw=2, c="y", label="w/(sqrt(2)rho) = 0.75")
plt.plot(np.sqrt(2)*rho_04 / w[4], Q_z4 , lw=2, c="b", label="w/(sqrt(2)rho) = 1")
plt.plot(np.sqrt(2)*rho_05 / w[5], Q_z5 , lw=2, c="m", label="w/(sqrt(2)rho) = 1.25")
#plt.plot(np.sqrt(2)*rho_06 / w[6], Q_z6 , lw=2, c="m", label="w/(sqrt(2)rho) = 2")




new_ticks1 = np.linspace(-3, 3, 7) # plot axis
print(new_ticks1)
plt.xticks(new_ticks1,fontsize=20)
plt.yticks(np.linspace(-1.2, 0, 5),fontsize=20)
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',-3))
ax.spines['bottom'].set_position(('data',0))

plt.legend(loc=4,fontsize=16)
plt.title('rho = 30um, w0 = 2um',fontsize=20)

plt.xlabel(r"$\rho_{0x}/(w/\sqrt{2})$",fontsize=20)
plt.ylabel('Qz',fontsize=20)
plt.grid()
plt.show()



