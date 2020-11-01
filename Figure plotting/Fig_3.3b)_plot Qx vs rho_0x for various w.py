# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 18:58:37 2020

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


import Module_Fx_stiffness_at_parameter as FTAP

import time




integration_method = 'manual'   # 'manual' or 'integrated'
grid_size = 200

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

P = 12.03  #optimal power required to levitate at w0 = 0.85um


##########################################
#FIG 3.3b) plot Qx vs rho_0x for various w
##########################################

#x-axis: radial offset rho_0x
#y-axis: Qx trapping efficiency

#key function: TQ.Fx_total_vs_rho0x_plot

rho_0 = [0,0]   



w = [w_0, 0.5 * np.sqrt(2)*rho, 0.55 * np.sqrt(2)*rho, 0.6 * np.sqrt(2)*rho, 0.75 * np.sqrt(2)*rho, np.sqrt(2)*rho, 1.25*np.sqrt(2)*rho, 1.5*np.sqrt(2)*rho]  #various beam radius

rho_00 = np.linspace(-4*w[0]/np.sqrt(2), 4*w[0]/np.sqrt(2), 100)   #x-axis offset range

rho_01 = np.linspace(-4*w[1]/np.sqrt(2), 4*w[1]/np.sqrt(2), 100)

rho_02 = np.linspace(-4*w[2]/np.sqrt(2), 4*w[2]/np.sqrt(2), 100)

rho_03 = np.linspace(-4*w[3]/np.sqrt(2), 4*w[3]/np.sqrt(2), 100)

rho_04 = np.linspace(-4*w[4]/np.sqrt(2), 4*w[4]/np.sqrt(2), 100)

rho_05 = np.linspace(-4*w[5]/np.sqrt(2), 4*w[5]/np.sqrt(2), 100)

rho_06 = np.linspace(-4*w[6]/np.sqrt(2), 4*w[6]/np.sqrt(2), 100)

rho_07 = np.linspace(-4*w[7]/np.sqrt(2), 4*w[7]/np.sqrt(2), 100)


radial_flistx0 = np.asarray(TQ.Fx_total_vs_rho0x_plot(rho_00,rho_0[1], rho, n_0, n_s, w_0, w[0], z_R, P, target = "reflective", integration_method = integration_method, grid_size = grid_size))

Q_x0 = radial_flistx0 * c / ( n_0 * P )


radial_flistx1 = np.asarray(TQ.Fx_total_vs_rho0x_plot(rho_01,rho_0[1], rho, n_0, n_s, w_0, w[1], z_R, P, target = "reflective", integration_method = integration_method, grid_size = grid_size))

Q_x1 = radial_flistx1* c / ( n_0 * P )

radial_flistx2 = np.asarray(TQ.Fx_total_vs_rho0x_plot(rho_02,rho_0[1], rho, n_0, n_s, w_0, w[2], z_R, P, target = "reflective", integration_method = integration_method, grid_size = grid_size))

Q_x2 = radial_flistx2 * c / ( n_0 * P )

radial_flistx3 = np.asarray(TQ.Fx_total_vs_rho0x_plot(rho_03,rho_0[1], rho, n_0, n_s, w_0, w[3], z_R, P, target = "reflective", integration_method = integration_method, grid_size = grid_size))

Q_x3 = radial_flistx3 * c / ( n_0 * P )

radial_flistx4 = np.asarray(TQ.Fx_total_vs_rho0x_plot(rho_04,rho_0[1], rho, n_0, n_s, w_0, w[4], z_R, P, target = "reflective", integration_method = integration_method, grid_size = grid_size))

Q_x4 = radial_flistx4* c / ( n_0 * P )

radial_flistx5 = np.asarray(TQ.Fx_total_vs_rho0x_plot(rho_05,rho_0[1], rho, n_0, n_s, w_0, w[5], z_R, P, target = "reflective", integration_method = integration_method, grid_size = grid_size))

Q_x5 = radial_flistx5 * c / ( n_0 * P )

radial_flistx6 = np.asarray(TQ.Fx_total_vs_rho0x_plot(rho_06,rho_0[1], rho, n_0, n_s, w_0, w[6], z_R, P, target = "reflective", integration_method = integration_method, grid_size = grid_size))

Q_x6 = radial_flistx6 * c / ( n_0 * P )

radial_flistx7 = np.asarray(TQ.Fx_total_vs_rho0x_plot(rho_07,rho_0[1], rho, n_0, n_s, w_0, w[7], z_R, P, target = "reflective", integration_method = integration_method, grid_size = grid_size))

Q_x7 = radial_flistx7 * c / ( n_0 * P )



plt.figure(11)
#plt.plot(np.sqrt(2)*rho_00 / w[0], Q_x0, lw=2, c="c", label="w = w0")
plt.plot(np.sqrt(2)*rho_01 / w[1], Q_x1, lw=2, c="r", label="w/(sqrt(2)rho) = 0.5")
#plt.plot(np.sqrt(2)*rho_02 / w[2], Q_x2, lw=2, c="g", label="w/(sqrt(2)rho) = 0.55")
plt.plot(np.sqrt(2)*rho_03 / w[3], Q_x3, lw=2, c="y", label="w/(sqrt(2)rho) = 0.6")
plt.plot(np.sqrt(2)*rho_04 / w[4], Q_x4, lw=2, c="b", label="w/(sqrt(2)rho) = 0.75")
plt.plot(np.sqrt(2)*rho_05 / w[5], Q_x5, lw=2, c="c", label="w/(sqrt(2)rho) = 1")
plt.plot(np.sqrt(2)*rho_06 / w[6], Q_x6, lw=2, c="m", label="w/(sqrt(2)rho) = 1.25")
#plt.plot(np.sqrt(2)*rho_07 / w[7], Q_x7, lw=2, c="c", label="w/(sqrt(2)rho) = 1.5")

new_ticks1 = np.linspace(-4, 4, 9) # plot axis
print(new_ticks1)
plt.xticks(new_ticks1,fontsize=20)
plt.yticks(np.linspace(-0.3, 0.3, 7),fontsize=20)

plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',-4))
ax.spines['bottom'].set_position(('data',0))

plt.legend(loc=1,fontsize=16)

plt.xlabel('rho_0x/(w/sqrt(2))',fontsize=20)
plt.ylabel('Qx',fontsize=20)

plt.title('rho = 30um, w0 = 2um',fontsize=20)
plt.grid()
plt.show()




