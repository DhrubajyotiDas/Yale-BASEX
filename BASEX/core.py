#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from time import time
import os.path

import numpy as np
from numpy.linalg import inv
from scipy.ndimage import median_filter, gaussian_filter, map_coordinates

from .basis import generate_basis_sets
from .io import parse_matlab


######################################################################
# PyBASEX - A Python BASEX implementation
# Dan Hickstein - University of Colorado Boulder
# danhickstein@gmail.com
#
# This is adapted from the BASEX Matlab code provided by the Reisler group.
#
# Please cite: "The Gaussian basis-set expansion Abel transform method"
# V. Dribinski, A. Ossadtchi, V. A. Mandelshtam, and H. Reisler,
# Review of Scientific Instruments 73, 2634 (2002).
#
# Version 0.4 - 2015-05-0
#   Major code update see pull request
#   https://github.com/DanHickstein/pyBASEX/pull/3
# Version 0.3 - 2015-02-01
#   Added documentation
# Version 0.2 - 2014-10-09
#   Adding a "center_and_transform" function to make things easier
# Versions 0.1 - 2012
#   First port to Python
#
#
# To-Do list:
#
#   I took all of the linear algebra straight from the Matlab program. It's
#   a little hard to compare with the Rev. Sci. Instrum. paper. It would be
#   nice to clean this up so that it's easier to follow along with the paper.
#
#   Currently, this program just uses the 1000x1000 basis set generated using
#   the Matlab implementation of BASEX. It would be good to port the basis set
#   generating functions as well. This would give people the flexibility to use
#   different sized basis sets. For example, some image may need higher resolution
#   than 1000x1000, or, transforming larger quantities of low-resolution images
#   may be faster with a 100x100 basis set.
#
########################################################################

class BASEX(object):

    def __init__(self, n_vert=1001, n_horz=501, 
                    nbf_vert = 1000, nbf_horz = 250,
                    basis_dir='./',
                    use_basis_set=None, verbose=True, calc_speeds=False):
        """ Initalize the BASEX class, preloading or generating the basis set.

        Parameters:
        -----------
          - n_vert: integer: number of vertical pixels in image to be analyzed
          - n_horz: ODD integer: number of horizontal pixels in image to be analyzed (MUST BE ODD)
          
          (Abel inverse transform will be performed on a 'n_vert x n_horz' area of the image)

          - nbf_vert: integer: number of basis functions to describe axial behavior
          - nbf_horz: integer: number of basis functions to describe radial behavior
          - basis_dir : path to the directory for saving / loading the basis set coefficients.
          - use_basis_set: use the basis set stored as a text files, if
                  it provided, the following parameters will be ignored 
                  n_vert, n_horz, nbf_vert, nbf_horz, basis_dir
                  The expected format is a string of the form "some_basis_set_{}_1.bsc" where 
                  "{}" will be replaced by "" for the first file and "pr" for the second.
                  Gzip compressed text files are accepted.
                  For instance, we would use:
                  use_basis_set_matlab="../BASEX/data/ascii/original_basis1000{}_1.txt.gz"
                  to load the basis set included with this package.
          - verbose: Set to True to see more output for debugging
          - calc_speeds: determines if the speed distribution should be calculated
        """
        n_horz = 2*(n_horz//2) + 1 # make sure n is odd

        self.verbose = verbose
        self.calc_speeds = calc_speeds

        self.n_vert = n_vert
        self.n_horz = n_horz
        self.nbf_vert = nbf_vert
        self.nbf_horz = nbf_horz

        if self.verbose:
            t1 = time()

        basis_name = "basex_basis_{}_{}_{}_{}.npy".format(n_vert, n_horz, nbf_vert, nbf_horz)
        path_to_basis_file = os.path.join(basis_dir, basis_name)
        
        if use_basis_set is not None:
            # load the matlab generated basis set
            M, Mc = parse_matlab(use_basis_set)
            left, right = get_left_right_matrices(M, Mc)

            self.n, self.nbf = M.shape # overwrite the provided parameters

        elif os.path.exists(path_to_basis_file):
            # load the basis set generated with this python module,
            # saved as a .npy file
            if self.verbose:
                print('Loading basis sets...           ')
            M_vert, M_horz, Mc_vert, Mc_horz = np.load(path_to_basis_file) 
            vert_left, horz_right = get_left_right_matrices(M_vert, M_horz, Mc_vert, Mc_horz)

        else:
            # generate the basis set
            if self.verbose:
                print('A suitable basis set was not found.\n',
                      'A new basis set will be generated.\n',
                      'This may take a few minutes.\n',
                      'But don\'t worry, it will be saved to disk for future use.\n')

            M_vert, M_horz, Mc_vert, Mc_horz  = generate_basis_sets(n_vert, n_horz, nbf_vert, nbf_horz, verbose=verbose)
            vert_left, horz_right = get_left_right_matrices(M_vert, M_horz, Mc_vert, Mc_horz)

            np.save(path_to_basis_file, (M_vert, M_horz, Mc_vert, Mc_horz))
            print('Basis set saved for later use to,')
            print(' '*10 + '{}'.format(path_to_basis_file))

        self.vert_left, self.horz_right = vert_left, horz_right
        self.M_vert, self.M_horz, self.Mc_vert, self.Mc_horz = M_vert, M_horz, Mc_vert, Mc_horz

        if self.verbose:
            print('{:.2f} seconds'.format((time()-t1)))


    def _basex_transform(self, rawdata):
        """ This is the core function that does the actual transform, 
            but it's not typically what is called by the user
         INPUTS:
          rawdata: a N_vert x N_horz numpy array of the raw image.
          verbose: Set to True to see more output for debugging
          calc_speeds: determines if the 1D speed distribution should be calculated (takes a little more time)

         RETURNS:
          IM: The abel-transformed image, a slice of the 3D distribution
          speeds: (optional) a array of length=500 of the 1D distribution, integrated over all angles
        """
        vert_left, horz_right = self.vert_left, self.horz_right
        Mc_vert, Mc_horz = self.Mc_vert, self.Mc_horz


        ### Reconstructing image  - This is where the magic happens ###
        if self.verbose:
            print('Reconstructing image...         ')
            t1 = time()

        Ci = vert_left.dot(rawdata).dot(horz_right)
        Ci = Ci * 1.1122244156826457 # Scaling factor necessary to match analytical Abel inverse
        # Haven't yet figured out why we need the scaling factor (?!?!)
        # P = dot(dot(Mc,Ci),M.T) # This calculates the projection, which should recreate the original image
        IM = Mc_vert.dot(Ci).dot(Mc_horz.T)

        if self.verbose:
            print('%.2f seconds' % (time()-t1))

        if self.calc_speeds:
            speeds = self.calculate_speeds(IM)
            return IM, speeds
        else:
            return IM


    def __call__(self, data, center_column,
                             median_size=0, gaussian_blur=0, post_median=0,
                             symmetrize=False, verbose=True):
        """ This is the main function that is called by the user. 
            It center the image, blurs the image (if desired)
            and completes the BASEX transform.

         Inputs:
         data - a N_vert x N_horz numpy array
                If N is smaller than the size of the basis set, zeros will be padded on the edges.
         center_column - the axis of symmetry of the image (column #)
         median_size - size (in pixels) of the median filter that will be applied to the image before
                       the transform. This is crucial for emiminating hot pixels and other
                       high-frequency sensor noise that would interfere with the transform
         gaussian_blur - the size (in pixels) of the gaussian blur applied before the BASEX tranform.
                         this is another way to blur the image before the transform.
                         It is normally not used, but if you are looking at very broad features
                         in very noisy data and wich to apply an aggressive (large radius) blur
                         (i.e., a blur in excess of a few pixels) then the gaussian blur will
                         provide better results than the median filter.
         post_median - this is the size (in pixels) of the median blur applied AFTER the BASEX transform
                       it is not normally used, but it can be a good way to get rid of high-frequency
                       artifacts in the transformed image. For example, it can reduce centerline noise.
         verbose - Set to True to see more output for debugging
         calc_speeds - determines if the speed distribution should be calculated
        """
        
        # make sure that the data is the right shape (1D must be converted to 2D)
        data = np.atleast_2d(data) # if passed a 1D array convert it to 2D
        self.verbose = verbose
        if data.shape[0] == 1:
            self.ndim = 1
        elif data.shape[1] == 1:
            raise ValueError('Wrong input shape for data {0}, should be  (N1, N2) or (1, N), not (N, 1)'.format(data.shape))
        else:
            self.ndim = 2

        image = self._center_image(data, center_column=center_column)

        if symmetrize:
            #image = apply_symmetry(image)
            raise NotImplementedError

        if median_size>0:
            image = median_filter(image,size=median_size)

        if gaussian_blur>0:
            image = gaussian_filter(image,sigma=gaussian_blur)

        #Do the actual transform
        res = self._basex_transform(image)

        if self.calc_speeds:
            recon, speeds = res
        else:
            recon = res

        if post_median > 0:
            recon = median_filter(recon, size=post_median)


        if self.ndim == 1:
            recon = recon[0, :] # taking one row, since they are all the same anyway

        if self.calc_speeds:
            return recon, speeds
        else:
            return recon

    def _center_image(self, data, center_column):
        n_vert, n_horz = self.n_vert, self.n_horz
        if data.ndim > 2:
            raise ValueError("Array to be centered must be 1- or 2-dimensional")

        c_im = np.copy(data) # make a copy of the original data for manipulation
        data_vert, data_horz = c_im.shape

        if data_horz % 2 == 0:
            # Add column of zeros to the extreme right to give data array odd columns
            c_im = np.lib.pad(c_im, ((0,0),(0,1)), 'constant', constant_values=0)
            data_vert, data_horz = c_im.shape # update data dimensions

        delta_h = int(center_column - data_horz//2)
        if delta_h != 0:
            if delta_h < 0: 
                # Specified center is to the left of nominal center
                # Add compensating zeroes on the left edge
                c_im = np.lib.pad(c_im, ((0,0),(2*np.abs(delta_h),0)), 'constant', constant_values=0)
                data_vert, data_horz = c_im.shape
            else:
                # Specified center is to the right of nominal center
                # Add compensating zeros on the right edge
                c_im = np.lib.pad(c_im, ((0,0),(0,2*delta_h)), 'constant', constant_values=0)
                data_vert, data_horz = c_im.shape

        if n_vert >= data_vert and n_horz >= data_horz:
            pad_up = (n_vert - data_vert)//2
            pad_down = n_vert - data_vert - pad_up
            pad_left = (n_horz - data_horz)//2
            pad_right = n_horz - data_horz - pad_left
            c_im = np.lib.pad(c_im, ((pad_up,pad_down), (pad_left,pad_right)), 'constant', constant_values=0)

        elif n_vert >= data_vert and n_horz < data_horz:
            pad_up = (n_vert - data_vert)//2
            pad_down = n_vert - data_vert - pad_up
            crop_left = (data_horz - n_horz)//2
            crop_right = data_horz - n_horz - crop_left
            if self.verbose:
                print("Warning: cropping %d pixels from the sides of the image" %crop_left)
            c_im = np.lib.pad(c_im[:,crop_left:-crop_right], ((pad_up, pad_down), (0,0)), 'constant', constant_values=0)

        elif n_vert < data_vert and n_horz >= data_horz:
            crop_up = (data_vert - n_vert)//2
            crop_down = data_vert - n_vert - crop_up
            pad_left = (n_horz - data_horz)//2
            pad_right = n_horz - data_horz - pad_left
            if self.verbose:
                print("Warning: cropping %d pixels from top and bottom of the image" %crop_up)
            c_im = np.lib.pad(c_im[crop_up:-crop_down], ((0,0), (pad_left, pad_right)), 'constant', constant_values=0)

        elif n_vert < data_vert and n_horz < data_horz:
            crop_up = (data_vert - n_vert)//2
            crop_down = data_vert - n_vert - crop_up
            crop_left = (data_horz - n_horz)//2
            crop_right = data_horz - n_horz - crop_left
            if self.verbose:
                print("Warning: cropping %d pixels from top and bottom and %d pixels from the sides of the image " %(crop_up, crop_left))
            c_im = c_im[crop_up:-crop_down,crop_left:-crop_right]

        else:
            raise ValueError('Input data dimensions incompatible with chosen basis set.')

        return c_im


    def calculate_speeds(self, IM):
        # This section is to get the speed distribution.
        # The original matlab version used an analytical formula to get the speed distribution directly
        # from the basis coefficients. But, the C version of BASEX uses a numerical method similar to
        # the one implemented here. The difference between the two methods is negligable.
        """ Generating the speed distribution """

        if self.verbose:
            print('Generating speed distribution...')
            t1 = time()

        nx,ny = np.shape(IM)
        xi = np.linspace(-100, 100, nx)
        yi = np.linspace(-100, 100, ny)
        X,Y = np.meshgrid(xi,yi)

        polarIM, ri, thetai = reproject_image_into_polar(IM)

        speeds = np.sum(polarIM, axis=1)
        speeds = speeds[:self.n//2] #Clip off the corners

        if self.verbose:
            print('%.2f seconds' % (time()-t1))
        return speeds


def center_image_old(data, center, n, ndim=2):
    """ This centers the image at the given center and makes it of size n by n"""
    
    Nh,Nw = data.shape
    n_2 = n//2
    if ndim == 1:
        cx = int(center)
        im = np.zeros((1,2*n))
        im[0, n-cx:n-cx+Nw] = data
        im = im[:, n_2:n+n_2]
        # This is really not efficient
        # Processing 2D image with identical rows while we just want a
        # 1D slice 
        im = np.repeat(im, n, axis=0)

    elif ndim == 2:
        cx, cy = np.asarray(center, dtype='int')
        
        #make an array of zeros that is large enough for cropping or padding:
        sz = 2*np.round(n+np.max((Nw,Nh)))
        im = np.zeros((sz,sz))
        im[sz//2-cy:sz//2-cy+Nh, sz//2-cx:sz//2-cx+Nw] = data
        im = im[ sz//2-n_2-1:n_2+sz//2, sz//2-n_2-1:n_2+sz//2] #not sure if this exactly preserves the center
        print(np.shape(im))
    else:
        raise ValueError

    return im

def get_left_right_matrices(M_vert, M_horz, Mc_vert, Mc_horz): 
    nbf_vert, nbf_horz = np.shape(M_vert)[1], np.shape(M_horz)[1]
    
    q_vert, q_horz = 0,0
    E_vert, E_horz = np.identity(nbf_vert)*q_vert, np.identity(nbf_horz)*q_horz

    vert_left = inv(Mc_vert.T.dot(Mc_vert) + E_vert).dot(Mc_vert.T)
    horz_right = M_horz.dot(inv(M_horz.T.dot(M_horz) + E_horz))

    return vert_left, horz_right


# I got these next two functions from a stackoverflow page and slightly modified them.
# http://stackoverflow.com/questions/3798333/image-information-along-a-polar-coordinate-system
# It is possible that there is a faster way to get the speed distribution.
# If you figure it out, pease let me know! (danhickstein@gmail.com)
def reproject_image_into_polar(data, origin=None):
    """Reprojects a 3D numpy array ("data") into a polar coordinate system.
    "origin" is a tuple of (x0, y0) and defaults to the center of the image.
    """
    ny, nx = data.shape[:2]
    if origin is None:
        origin = (nx//2, ny//2)

    # Determine that the min and max r and theta coords will be...
    x, y = index_coords(data, origin=origin)
    r, theta = cart2polar(x, y)

    nr = r.max()
    nt = ny//2

    # Make a regular (in polar space) grid based on the min and max r & theta
    r_i = np.linspace(r.min(), r.max(), nr)
    theta_i = np.linspace(theta.min(), theta.max(), nt)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    # Project the r and theta grid back into pixel coordinates
    X, Y = polar2cart(r_grid, theta_grid)
    X += origin[0] # We need to shift the origin
    Y += origin[1] # back to the lower-left corner...
    xi, yi = X.flatten(), Y.flatten()
    coords = np.vstack((xi,yi)) # (map_coordinates requires a 2xn array)

    zi = map_coordinates(data, coords)
    output = zi.reshape((nr,nt))
    return output, r_i, theta_i


def index_coords(data, origin=None):
    """Creates x & y coords for the indicies in a numpy array "data".
    "origin" defaults to the center of the image. Specify origin=(0,0)
    to set the origin to the lower left corner of the image.
    """
    ny, nx = data.shape[:2]
    if origin is None:
        origin_x, origin_y = nx // 2, ny // 2
    else:
        origin_x, origin_y = origin
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x -= origin_x
    y -= origin_y
    return x, y

def cart2polar(x, y):
    """
    Transform carthesian coordinates to polar
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y,x)
    return r, theta

def polar2cart(r, theta):
    """
    Transform polar coordinates to carthesian
    """
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    return x, y
