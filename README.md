# Yale-BASEX

Yale-BASEX is a Python implementation of the BASEX algorithm created by Dribinski, Ossadtchi, Mandelshtam, and Reisler [Rev. Sci. Instrum. 73 2634, (2002)]. This project is forked from PyAbel's python implementation of the BASEX algorithm, [found here.](https://github.com/PyAbel/PyAbel) The primary difference between Yale-BASEX and PyAbel is that Yale-BASEX is stripped down version of PyAbel focused exclusively on analyzing images of **coflow flames**, which do not have the *I(r,-z) = I(r,z)* symmetry. It does not have all the bells and whistles of the full blown PyAbel program, nor does it implement any other Abel inversion algorithms besides BASEX.

The main function of this program is to perform the inverse Abel transform on a two-dimensional flame image (which is assumed to be cylindrically symmetric around a vertical axis at *r* = 0). The inverse Abel transform takes a 2D projection of a cylindrically symmetric 3D image and return the central slice (2D) of the 3D distribution. The BASEX implementation uses modified Gaussian basis functions to find the transform instead of analytically solving the inverse Abel transform or applying the Fourier-Hankel method, as both the analytical solution and the Fourier-Hankel methods provide lower quality transforms when applied to real-world datasets (see the RSI paper). 

In this code, the axis of cylindrical symmetry is in assumed to be in the vertical direction. If this is not the case for your data, the numpy.rot90 function may be useful.

### Installation notes

To install this module run,

    python setup.py install --user
	
To tinker with the module yourself, run
	
	python setup.py develop

### Example of use

	from BASEX import BASEX
	import matplotlib.pyplot as plt
	import scipy as sc

	# Import example flame image
	filename = 'examples/data/flames/raw_flame.dat' 
	raw_data = sc.genfromtxt(filename)

	# Specify the center column of the image
	center = 191

	print('Performing the inverse Abel transform:')

	# Load (or pre-calculate if needed) the basis set for a 1601x383 image
	inv_ab = BASEX(n_vert = 1601, n_horz = 383,
               	nbf_vert = 1601, nbf_horz = 191, 
               	basis_dir = './', verbose = True)

    # Calculate the inverse Abel transform for the centered data
    recon = inv_ab(raw_data, center)

    # Plot the inverted images
    plt.imshow(recon)
    plt.show()

Have fun!
