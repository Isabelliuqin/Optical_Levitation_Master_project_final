# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 10:26:05 2020

@author: liuqi
"""


"""Plot overlaying diagram"""

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




#Our sphere

g = 9.8 

c = 3 * 10**8

w_0 = 2.67 * 10 ** (-6)

Lambda = 1.064 * 10**(-6)

z_R = np.pi* w_0 ** 2 / Lambda

rho = 30 * 10 ** (-6)
 
n_0 = 1

n_s_n = 0.04

k = 7.6097

n_s = n_s_n - k*1j

sig_s = 10.49 * 10 ** 3 * ( 3 ** 3 - 2.9 ** 3 / 3 ** 3 ) #density of sphere in kg/m^3

sig_0 = 0 #density of medium in kg/m^3

m = 4/3 * np.pi * rho ** 3 * ( sig_s - sig_0 )

Permittivity = 8.85 * 10**(-12)

P = 0.5 * c * n_0 * Permittivity    #total power of the LG01 beam normalised



#parameters
rho_0 = [15 * 10 ** (-6), 0]
w = 20 * 10 ** (-6)



#plotting setup
gridsize = 400

theta = np.linspace(0, np.pi/2, gridsize)
phi = np.linspace(0, np.pi * 2, gridsize)

Theta,Phi = np.meshgrid(theta, phi)


x = rho * np.sin(Theta) * np.cos(Phi)

y = rho * np.sin(Theta) * np.sin(Phi)



r = np.sqrt( (rho * np.sin(Theta) * np.cos(Phi) - rho_0[0]) ** 2 + (rho * np.sin(Theta) * np.sin(Phi) - rho_0[1]) ** 2) #represent r(the parameter of laser beam) by theta



I = TQ.LG_01_Intensity_sphere_coordinate(Theta, Phi, rho_0[0], rho_0[1], rho, w, n_0, P)





circle1 = plt.Circle((0, 0), rho * 10 ** 6, color='k', fill=False)

plt.gcf().gca().add_artist(circle1)

plt.contourf(x * 10 ** 6, y * 10 ** 6, I/np.max(I), 500, cmap='Reds')
cbar = plt.colorbar()

cbar.set_ticks([0,0.5, 1])

plt.xlabel('x(um)',fontsize=60)
plt.ylabel('y(um)',fontsize=60)
plt.rc('xtick',labelsize=60)
plt.rc('ytick',labelsize=60)
plt.xlim(-30, 30)
plt.ylim(30, -30)
