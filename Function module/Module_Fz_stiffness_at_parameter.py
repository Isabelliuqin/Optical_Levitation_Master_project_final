# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 15:31:40 2020

@author: liuqi
"""


#################################################
#axial stiffness at particular parameter functions
#################################################

import scipy as sp
import numpy as np
import matplotlib.pylab as plt
import scipy.integrate as spi
from scipy.integrate import quad
import seaborn
from scipy.integrate import odeint
from scipy.integrate import dblquad
import cmath
from sympy import *
from scipy.special import eval_genlaguerre as LGpoly
#import Module_Gauthier_result_functions as GRF
from scipy.misc import derivative
import Will_Module_addwdep as WMTQ

def Fz_stiffness_at_parameter(rho_0x,rho_0y, rho, n_0, n_s, w_0, w, z_R, P, resolution, target, integration_method, grid_size = 500):
    """Compute dFz/drho_0z at a chosen rho_0z with a resolution, need to take the average of the output array"""
    
    # rho_0x - beam x displacement, SCALAR
    # rho_0y - beam y displacement, scalar
    # rho - sphere radius, scalar
    # n_0 - refractive index of medium, scalar
    # n_s - refractive index of sphere, scalar
    # w_0 - beam radius at waist, scalar
    # w - beam radius at target bottom, scalar
    # z_R - beam rayleigh range, scalar
    # P - beam power (Watts), scalar
    # target - string, set as 'reflective' if sphere is reflective, otherwise anything OK and will be ignored
    
    # integration_method - str, 'integrated' or 'manual', use integrated or manual integration method
    # grid_size - integer, number of grids
    
    #resolution:  0.01 * 10 ** (-6) good enough
    chosen_resolution = resolution
    w_minus_2 = w - 2 * resolution
    w_minus_1 = w - 1 * resolution
    
    w_plus_1 = w + 1 * resolution
    w_plus_2 = w + 2 * resolution
    
    w_input = np.asarray([w_minus_2, w_minus_1, w, w_plus_1, w_plus_2])
    
    Fz_grad = WMTQ.Fz_total_gradient(rho_0x, rho_0y, rho, n_0, n_s, w_0, w_input, z_R, P, target, integration_method, grid_size)
    Fz_grad_output = np.asarray(Fz_grad['Fz_grad'])
    
    
    return Fz_grad_output


def Fz_stiffness_vs_w0_plots(rho_0x,rho_0y, rho, n_0, n_s, w_0, w, z_R, P, resolution, target, integration_method, grid_size = 500):
    """Compute dFz/drho_0z at a chosen rho_0z at various beam waist w0"""
    
    # rho_0x - beam x displacement, SCALAR
    # rho_0y - beam y displacement, scalar
    # rho - sphere radius, scalar
    # n_0 - refractive index of medium, scalar
    # n_s - refractive index of sphere, scalar
    # w_0 - beam radius at waist, list
    # w - beam radius at target bottom, scalar
    # z_R - beam rayleigh range, scalar
    # P - beam power (Watts), scalar
    # target - string, set as 'reflective' if sphere is reflective, otherwise anything OK and will be ignored
    
    # integration_method - str, 'integrated' or 'manual', use integrated or manual integration method
    # grid_size - integer, number of grids
    
    #resolution:  0.01 * 10 ** (-6) good enough
    
    Fz_grad = []
    
    Lambda = 1.064 * 10**(-6)
    
    for w0_e in w_0:
        
        z_Re = np.pi* w0_e ** 2 / Lambda
        Fz_grad_element = np.mean(Fz_stiffness_at_parameter(rho_0x,rho_0y, rho, n_0, n_s, w0_e, w, z_Re, P, resolution, target, integration_method, grid_size))
        
        Fz_grad.append(Fz_grad_element)
        
    return Fz_grad

def Fz_rho_0x_stiffness_at_parameter(rho_0x,rho_0y, rho, n_0, n_s, w_0, w, z_R, P, resolution, target, integration_method, grid_size = 500):
    """Compute dFz/drho_0x at a chosen rho_0x with a resolution, need to take the average of the output array"""
    # rho_0x - beam x displacement, SCALAR
    # rho_0y - beam y displacement, scalar
    # rho - sphere radius, scalar
    # n_0 - refractive index of medium, scalar
    # n_s - refractive index of sphere, scalar
    # w_0 - beam radius at waist, scalar
    # w - beam radius at target bottom, scalar
    # z_R - beam rayleigh range, scalar
    # P - beam power (Watts), scalar
    # target - string, set as 'reflective' if sphere is reflective, otherwise anything OK and will be ignored
    
    # integration_method - str, 'integrated' or 'manual', use integrated or manual integration method
    # grid_size - integer, number of grids
    
    #resolution:  0.01 * 10 ** (-6) good enough
    
    chosen_resolution = resolution
    rho0x_minus_2 = rho_0x - 2 * resolution       #expand the scalar rho_0x into a list around the chosen rho_0x
    rho0x_minus_1 = rho_0x - 1 * resolution
    
    rho0x_plus_1 = rho_0x + 1 * resolution
    rho0x_plus_2 = rho_0x + 2 * resolution
    
    rho0x_input = np.asarray([rho0x_minus_2, rho0x_minus_1, rho_0x, rho0x_plus_1, rho0x_plus_2])
    
    Fz_grad = WMTQ.Fz_total_gradient_vs_radialoffset(rho0x_input,rho_0y, rho, n_0, n_s, w_0, w, z_R, P, target, integration_method, grid_size)
    Fz_grad_output = np.asarray(Fz_grad['Fz_grad'])
    
    
    return Fz_grad_output


