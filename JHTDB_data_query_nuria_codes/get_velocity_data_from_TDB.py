import ctypes
import math
import os
import sys

import numpy
import pyJHTDB
import pyJHTDB.dbinfo
import pyJHTDB.interpolator

if pyJHTDB.found_matplotlib:
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
else:
    print('matplotlib is needed for contour plots.'
        + 'You should be able to find installation instructions at http://matplotlib.sourceforge.net')

if pyJHTDB.found_h5py:
    import h5py
else:
    print('h5py is needed for working with cutouts.')

def get_velocity_data(x_min=30.2185, x_max=1000.065, y_min=0, y_max=26.488, z_min=0, z_max=240, nx=831, ny=280, nz=512, time=0.25):
    
    # Time and spatial coordinates
    x = numpy.linspace(x_min, x_max, nx, dtype = 'float32')
    y = numpy.linspace(y_min, y_max, ny, dtype = 'float32')
    z = numpy.linspace(z_min, z_max, nz, dtype = 'float32')
    coord = numpy.empty((nx, ny, nz, 3), dtype = 'float32') # done for having all values at same resolution
    coord[:, :, :, 0] = x[:, numpy.newaxis, numpy.newaxis]
    coord[:, :, :, 1] = y[numpy.newaxis, :, numpy.newaxis]
    coord[:, :, :, 2] = z[numpy.newaxis, numpy.newaxis, :]
    
    # Interpolation configurations
    spatialInterp  = 4  # 4 point Lagrange
    temporalInterp = 0  # no time interpolation
    FD4Lag4        = 40 # 4 point Lagrange interp for derivatives
    
    # Print coordinates (where you will get the field values)
    print('Coordinates of {0}x{1}x{2} points where variables are requested:'.format(nx,ny,nz))
    for p in range(2):
        for q in range(2):
            for r in range(2):
                print('index {0}: {1}'.format([p,q,r], coord[p,q,r]))
    print('Data is requested at time {0}'.format(time))
    
    # To get data from a shared library, it is required to:
    # load shared library
    lTDB = pyJHTDB.libJHTDB()
    #initialize webservices
    lTDB.initialize()
    
    # Get velocity
    print('\n Requesting velocity at {0}x{1}x{2} points...'.format(nx,ny,nz))
    vel = lTDB.getData(time, coord,
            sinterp = spatialInterp, tinterp = temporalInterp,
            getFunction = 'getVelocity')
    for p in range(2):
        for q in range(2):
            for r in range(2):
                print('index {0}: {1}'.format([p,q,r], vel[p,q,r]))
        
    # Get velocity gradient
    print('\n Requesting velocity gradient at {0}x{1}x{2} points...'.format(nx,ny,nz))
    vel_grad = lTDB.getData(time, coord,
            sinterp = FD4Lag4, tinterp = temporalInterp,
            getFunction = 'getVelocityGradient')
    for p in range(2):
        for q in range(2):
            for r in range(2):
                print('index {0}: '.format([p,q,r]) +
                      'duxdx = {0:+e}, duxdy = {1:+e}, duxdz = {2:+e}\n   '.format(vel_grad[p,q,r][0], vel_grad[p,q,r][1], 
                                                                                   vel_grad[p,q,r][2]) +
                      '              duydx = {0:+e}, duydy = {1:+e}, duydz = {2:+e}\n   '.format(vel_grad[p,q,r][3], vel_grad[p,q,r][4],
                                                                                                 vel_grad[p,q,r][5]) +
                      '              duzdx = {0:+e}, duzdy = {1:+e}, duzdz = {2:+e}'.format(vel_grad[p,q,r][6], vel_grad[p,q,r][7],
                                                                                            vel_grad[p,q,r][8]))
        
    # Save data in h5 file:
    file_name = "JHTDB_time_"+str(time).replace('.','-')+"_n_"+str(nx)+"x"+str(ny)+"x"+str(nz)+".h5"
    file_path = "../../../Temporary/nuria_masclans/scratch/" + file_name
    with h5py.File(file_path, "w") as f:
        f.attrs.create('time',time)
        f.attrs.create('nx',nx)
        f.attrs.create('ny',ny)
        f.attrs.create('nz',nz)
        f.create_dataset('coord',data=coord)
        f.create_dataset('vel',data=vel)
        f.create_dataset('vel_grad',data=vel_grad)
    print("\nDatabase file created: " + file_path)
    
    #finalize webservices
    lTDB.finalize()  
    return None
                      
def get_velocity_data_test(N):
    
    # Time and spatial coordinates
    # time = 0.002 * numpy.random.randint(1024)
    time = 0.25
    coord = numpy.empty((N, 3), dtype = 'float32')
    coord[:,:] = 2*math.pi*numpy.random.random_sample(size = (N, 3))[:,:]
    
    # Interpolation configurations
    spatialInterp  = 6  # 6 point Lagrange
    temporalInterp = 0  # no time interpolation
    FD4Lag4        = 40 # 4 point Lagrange interp for derivatives
    
    # Print coordinates (where you will get the field values)
    print('Coordinates of {0} points where variables are requested:'.format(N))
    for p in range(N):
        print('{0}: {1}'.format(p, coord[p]))
    print('Data is requested at time {0}'.format(time))
    
    # To get data from a shared library, it is required to:
    # load shared library
    lTDB = pyJHTDB.libJHTDB()
    #initialize webservices
    lTDB.initialize()
    
    # Get velocity
    print('Requesting velocity at {0} points...'.format(N))
    result = lTDB.getData(time, coord,
            sinterp = spatialInterp, tinterp = temporalInterp,
            getFunction = 'getVelocity')
    for p in range(N):
        print('{0}: {1}'.format(p, result[p]))
        
    # Get velocity gradient
    print('Requesting velocity gradient at {0} points...'.format(N))
    result = lTDB.getData(time, coord,
            sinterp = FD4Lag4, tinterp = temporalInterp,
            getFunction = 'getVelocityGradient')
    for p in range(N):
        print('{0}: '.format(p) +
              'duxdx = {0:+e}, duxdy = {1:+e}, duxdz = {2:+e}\n   '.format(result[p][0], result[p][1], result[p][2]) +
              'duydx = {0:+e}, duydy = {1:+e}, duydz = {2:+e}\n   '.format(result[p][3], result[p][4], result[p][5]) +
              'duzdx = {0:+e}, duzdy = {1:+e}, duzdz = {2:+e}'.format(result[p][6], result[p][7], result[p][8]))
        
    #finalize webservices
    lTDB.finalize()
    return None

def make_countours_ken_vel(levels=30, time=0.25, xmin=0.0, xmax=1**(-10), yoff=0.5, zmin=0.0, zmax=1**(-10), nx=64, nz=64, spatialInterp=4, temporalInterp=0):
    
    lTDB = pyJHTDB.libJHTDB()
    lTDB.initialize()
    
    # Plot kinetic energy contours
    ##  only if matplotlib is present
    if pyJHTDB.found_matplotlib:
        ken_contours('kinetic_energy_contours', lTDB, levels, spatialInterp, temporalInterp, time, xmin, xmax, yoff, zmin, zmax, nx, nz)
        
    # Plot instantaneous velocity contours, for each component ux, uy, uz
    if pyJHTDB.found_matplotlib:
        vel_contours('velocity_contours', lTDB, levels, spatialInterp, temporalInterp, time, xmin, xmax, yoff, zmin, zmax, nx, nz)
    
    # Plot instantaneous velocity gradient contours, components du/dy and dw/dy
    if pyJHTDB.found_matplotlib:
        vel_grad_contours('velocity_gradient_contours', lTDB, levels, spatialInterp, temporalInterp, time, xmin, xmax, yoff, zmin, zmax, nx, nz)
    
    lTDB.finalize()
    return None

def ken_contours(figname, lTDB, levels=30, spatialInterp=4, temporalInterp=0, time=0.25, xmin =0.0, xmax= 1**(-10), yoff=0.5, zmin=0.0,  zmax=1**(-10), nx=64, nz=64):
    """
        Generate a simple contour plot
        see http://matplotlib.sourceforge.net/examples/pylab_examples/contour_demo.html
        for information on how to make prettier plots.

        This function assumes the webservices have already been initialized,
        so call pyJHTDB.init() before calling it, and pyJHTDB.finalize() afterwards
    """
    x = numpy.linspace(xmin, xmax, nx, dtype = 'float32')
    z = numpy.linspace(zmin, zmax, nz, dtype = 'float32')
    coord = numpy.empty((nx, nz, 3), dtype = 'float32')
    coord[:, :, 0] = x[:, numpy.newaxis]
    coord[:, :, 1] = yoff
    coord[:, :, 2] = z[numpy.newaxis, :]
    
    # Get data kinetic energy
    result = lTDB.getData(time, coord,
            sinterp = spatialInterp, tinterp = temporalInterp,
            getFunction = 'getVelocity')
    energy = .5*(numpy.sqrt(result[:,:,0]**2 + result[:,:,1]**2 + result[:,:,2]**2)).transpose()
    
    # Make contour plot
    fig = plt.figure(figsize=(6.,6.))
    ax = fig.add_axes([.0, .0, 1., 1.])
    contour = ax.contour(x, z, energy, levels)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    plt.clabel(contour, inline=1, fontsize=10)
    plt.title('Energy contours, t = {0:.5}, y = {1:.3}'.format(time, yoff))
    fig.savefig(figname + '.eps', format = 'eps', bbox_inches = 'tight')
    return None
    

def vel_contours(figname, lTDB, levels=30, spatialInterp=4, temporalInterp=0, time=0.25, xmin =0.0, xmax= 1**(-10), yoff=0.5, zmin=0.0,  zmax=1**(-10), nx=64, nz=64):
    x = numpy.linspace(xmin, xmax, nx, dtype = 'float32')
    z = numpy.linspace(zmin, zmax, nz, dtype = 'float32')
    coord = numpy.empty((nx, nz, 3), dtype = 'float32')
    coord[:, :, 0] = x[:, numpy.newaxis]
    coord[:, :, 1] = yoff
    coord[:, :, 2] = z[numpy.newaxis, :]

    # Get data velocity components
    result = lTDB.getData(time, coord,
            sinterp = spatialInterp, tinterp = temporalInterp,
            getFunction = 'getVelocity')
    ux = result[:,:,0]
    uy = result[:,:,1]
    uz = result[:,:,2]
    
    # Contour plot ux
    fig = plt.figure(figsize=(6.,6.))
    ax = fig.add_axes([.0, .0, 1., 1.])
    contour = ax.contour(x, z, ux, levels)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    plt.clabel(contour, inline=1, fontsize=10)
    plt.title('Velocity ux contours, t = {0:.5}, y = {1:.3}'.format(time, yoff))
    fig.savefig(figname + '_ux.eps', format = 'eps', bbox_inches = 'tight')
    
    # Contour plot uy
    fig = plt.figure(figsize=(6.,6.))
    ax = fig.add_axes([.0, .0, 1., 1.])
    contour = ax.contour(x, z, uy, levels)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    plt.clabel(contour, inline=1, fontsize=10)
    plt.title('Velocity uy contours, t = {0:.5}, y = {1:.3}'.format(time, yoff))
    fig.savefig(figname + '_uy.eps', format = 'eps', bbox_inches = 'tight')
    
    # Contour plot uz
    fig = plt.figure(figsize=(6.,6.))
    ax = fig.add_axes([.0, .0, 1., 1.])
    contour = ax.contour(x, z, uz, levels)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    plt.clabel(contour, inline=1, fontsize=10)
    plt.title('Velocity uz contours, t = {0:.5}, y = {1:.3}'.format(time, yoff))
    fig.savefig(figname + '_uz.eps', format = 'eps', bbox_inches = 'tight')
    return None

def vel_grad_contours(figname, lTDB, levels=30, spatialInterp=4, temporalInterp=0, time=0.25, xmin =0.0, xmax= 1**(-10), yoff=0.5, zmin=0.0,  zmax=1**(-10), nx=64, nz=64):
    x = numpy.linspace(xmin, xmax, nx, dtype = 'float32')
    z = numpy.linspace(zmin, zmax, nz, dtype = 'float32')
    coord = numpy.empty((nx, nz, 3), dtype = 'float32')
    coord[:, :, 0] = x[:, numpy.newaxis]
    coord[:, :, 1] = yoff
    coord[:, :, 2] = z[numpy.newaxis, :]

    # Get data velocity components
    FD4Lag4        = 40
    result = lTDB.getData(time, coord,
            sinterp = FD4Lag4, tinterp = temporalInterp,
            getFunction = 'getVelocityGradient')
    dudy = result[:,:,1]
    dwdy = result[:,:,7]
    
    # Contour plot du/dy
    fig = plt.figure(figsize=(6.,6.))
    ax = fig.add_axes([.0, .0, 1., 1.])
    contour = ax.contour(x, z, dudy, levels)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    plt.clabel(contour, inline=1, fontsize=10)
    plt.title('Velocity Gradient du/dy contours, t = {0:.5}, y = {1:.3}'.format(time, yoff))
    fig.savefig(figname + '_dudy.eps', format = 'eps', bbox_inches = 'tight')
    
    # Contour plot dw/dy
    fig = plt.figure(figsize=(6.,6.))
    ax = fig.add_axes([.0, .0, 1., 1.])
    contour = ax.contour(x, z, dwdy, levels)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    plt.clabel(contour, inline=1, fontsize=10)
    plt.title('Velocity Gradient dw/dy contours, t = {0:.5}, y = {1:.3}'.format(time, yoff))
    fig.savefig(figname + '_dwdy.eps', format = 'eps', bbox_inches = 'tight')
    return None
    
if __name__ == '__main__':
    N = 10
    get_velocity_data_test(N)
    make_countours_ken_vel()
