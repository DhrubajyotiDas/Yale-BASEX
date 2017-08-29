#!/usr/bin/pthon
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from math import exp, log

import numpy as np
from scipy.special import gammaln
import sys

MAX_OFFSET = 4000

def generate_basis_sets(n_vert=1001, n_horz = 501, 
						nbf_vert = 1000, nbf_horz = 250, 
						verbose=True):
    """ 
    Generate the basis set for the BASEX method. 

    This function was adapted from the matlab script BASIS2.m and 
    the python script basis.py from Dan Hickstein's pyBASEX implementation
    with some optimizations.

    Parameters:
    -----------
      n_vert : integer : Vertical dimensions of the image in pixels. 
      n_horz : integer : Horizontal dimensions of the image in pixels. Must be odd.
      nbf_vert : integer : Number of basis functions in the z-direction. Must be less than or equal to n_vert
      nbf_horz: integer: Number of basis functions in the x-direction. Must be less than (n_horz + 1)/2 

    Returns:
    --------
      M_vert, M_horz, Mc_vert, Mc_horz : np.matrix
    """
    if n_horz % 2 == 0:
        raise ValueError('The n_horz parameter must be odd.')

    if (n_horz+1)//2 < nbf_horz:
        raise ValueError('The number of basis functions nbf_horz cannot be greater than (n_horz + 1)/2')

    if n_vert < nbf_vert:
    	raise ValueError('The number of basis functions nbf_vert cannot be greater than the number of vertical pixels n_vert.')

    Rm_h = n_horz//2 + 1

    I_h = np.arange(1, n_horz + 1)

    R2_h = (I_h - Rm_h)**2
    M_horz = np.zeros((n_horz, nbf_horz))
    Mc_horz = np.zeros((n_horz, nbf_horz))

    M_horz[:,0] = 2*np.exp(-R2_h)
    Mc_horz[:,0] = np.exp(-R2_h)

    gammaln_0o5 = gammaln(0.5) 

    if verbose:
        print('Generating horizontal BASEX basis sets for n_horz = {}, nbf_horz = {}:\n'.format(n_horz, nbf_horz))
        sys.stdout.write('0')
        sys.stdout.flush()

    # the number of elements used to calculate the projected coefficeints
    delta = np.fmax(np.arange(nbf_horz)*32 - 4000, 4000) 
    for k in range(1, nbf_horz):
        k2 = k*k # so we don't recalculate it all the time
        log_k2 = log(k2) 
        angn = exp(
                    k2 * (1 - log_k2) + 
                    gammaln(k2 + 0.5) - gammaln_0o5  
                    # old form --> k2 - 2 * k2*log(k) +
                    #              np.log(np.arange(0.5, k2 + 0.5)).sum()
                    )
        M_horz[Rm_h-1, k] =  2 * angn

        for l in range(1, n_horz - Rm_h + 1):
            l2 = l*l
            log_l2 = log(l2)

            val = exp(k2 * (1 + log(l2/k2)) - l2) 
            # old form --> val = exp(k2 - l2 + 2 * k2*log((1.0 * l)/k)) 

            Mc_horz[Rm_h - 1 + l, k] = val # All rows below center
            Mc_horz[Rm_h - 1 - l, k] = val # All rows above center

            aux = val + angn * Mc_horz[Rm_h - 1 + l, 0]

            p = np.arange(max(1, l2 - delta[k]), min(k2 - 1,  l2 + delta[k]) + 1)

            # We use here the fact that for p, k real and positive
            #
            #  np.log(np.arange(p, k)).sum() == gammaln(k) - gammaln(p) 
            #
            # where gammaln is scipy.misc.gammaln (i.e. the log of the Gamma function)
            #
            # The following line corresponds to the vectorized third
            # loop of the original BASIS2.m matlab file.

            aux += np.exp(k2 - l2 - k2*log_k2 + p*log_l2
                      + gammaln(k2 + 1) - gammaln(p + 1) 
                      + gammaln(k2 - p + 0.5) - gammaln_0o5
                      - gammaln(k2 - p + 1)
                      ).sum()

            # End of vectorized third loop

            aux *= 2

            M_horz[Rm_h - 1 + l, k] = aux # All rows below center
            M_horz[Rm_h - 1 - l, k] = aux # All rows above center

        if verbose and k % 50 == 0:
            sys.stdout.write('...{}'.format(k))
            sys.stdout.flush()

    if verbose:
        print("...{}".format(k+1))

    ####################################    
	# Axial functions
    ####################################

    Z2_h = np.arange(0, n_vert)**2

    M_vert = np.zeros((n_vert, nbf_vert))
    Mc_vert = np.zeros((n_vert, nbf_vert))

    # M_vert[:,0] = 2*np.exp(-Z2_h)
    Mc_vert[:,0] = np.exp(-Z2_h)

    if verbose:
        print('Generating vertical BASEX basis sets for n_vert = {}, nbf_vert = {}:\n'.format(n_vert, nbf_vert))
        sys.stdout.write('0')
        sys.stdout.flush()

    # delta_v = np.fmax(np.arange(nbf_vert)*32 - 0, 8000) 

    for k in range(1, nbf_vert):
    	k2 = k*k
    	log_k2 = log(k2)

    	for l in range(1, n_vert):
    		l2 = l*l
    		log_l2 = log(l2)

    		val = exp(k2 * (1 + log(l2/k2)) - l2) 

    		Mc_vert[l, k] = val 

    	if verbose and k % 50 == 0:
            sys.stdout.write('...{}'.format(k))
            sys.stdout.flush()

    if verbose:
        print("...{}".format(k+1))

    return M_vert, M_horz, Mc_vert, Mc_horz 

