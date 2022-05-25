import h5py
import matplotlib.pyplot as plt
import numpy as np


def get_averaged_profiles_JHTDB():
    '''
    Returns
    -------
    x : numpy.ndarray, shape (3320,)
        x-coordinates of the grid where averaged profiles are given
    y : numpy.ndarray, shape (224,)
        y-coordinates of the grid where averaged profiles are given
    um : numpy.ndarray, shape (224, 3320)
        x-velocity  averaged on time and on z-coordinate (<U>)
    vm : numpy.ndarray, shape (224, 3320)
        y-velocity averaged on time and on z-coordinate  (<V>)
    wm : numpy.ndarray, shape (224, 3320)
        z-velocity  averaged on time and on z-coordinate (<W>)
    '''
    file_path = "JHTDB_h5_files/Transition_BL_Time_Averaged_Profiles.h5"
    with h5py.File(file_path, "r") as f:
        x = f['x_coor'][:]
        y = f['y_coor'][:]
        um = f['um'][:] # <U>(x,y), averaged on time and on z-coordinate (as z-coord has no statistic influence in boundary layer, Pope ch. 7.3)
    return x,y,um

def transform_averaged_profile_to_wall_units(x,y,u_avg,nu):
    '''
    Parameters
    ----------
    x       : np.ndarray(3320,)         : x-coord of averaged profiles grid
    y       : np.ndarray(224,)          : y-coord of averaged profiles grid
    u_avg   : np.ndarray(224,3320)      : u-velocity (x-component) averaged in time and in z-coordinate
    nu      : float                     : fluid viscosity
    
    Returns
    -------
    y_plus  : np.ndarray(224 (y) ,3320 (x)) : y-coord in wall units, viscous length to the wall
    u_plus  : np.ndarray(224 (y) ,3320 (x)) : u-velocity (x-comp. of vel.) in wall units
    u_tau   : np.ndarray(3320,)             : friction velocity
    delta_v : np.ndarray(3320,)             : viscous lengthscale
    
    '''
    dy = y[0]-0                         # distance wall (y=0) to first grid point (y[0] = 0.00357...)
    du_avg_wall = u_avg[0,:]-0          # U_avg[0,:] in y[0]=0.0036, and U_avg_wall = 0 in y=0
    du_avg_wall_dy = du_avg_wall/dy 

    # Calculate Friction Velocity: u_tau = tau_w / rho, where tau_w is the wall shear stress (all viscous) and rho the density
    u_tau = np.sqrt(nu*du_avg_wall_dy)  # function of x-coord, shape (3320,)
    # Calculate Viscous Lengthscale
    delta_v = nu/u_tau                  # funciton of x-coord, shape (3320,)
    
    # Transform Grid of Averaged Profiles in wall-units (for y-coord (y-plus) and u (u-plus))
    X, Y = np.meshgrid(x,y)             # Grid of Averaged  Profiles, X, Y shape (240 (y-coor), 3320 (x-coor))
    y_plus = Y/delta_v                  # y-wall-coordinate, func. (x,y), shape (3320,224)
    u_plus = u_avg/u_tau                # u-wall-coordinate, func. (x,y), shape (3320,224)
    
    return y_plus, u_plus, u_tau, delta_v

def plot_Yplus_Uplus_relation(y_plus,u_plus,x_coord):
    plt.figure()
    plt.plot(y_plus[:,0],u_plus[:,0],label='x={}'.format(x_coord[0]))
    plt.plot(y_plus[:,int(len(x_coord)/2)],u_plus[:,int(len(x_coord)/2)],label='x={}'.format(x_coord[int(len(x_coord)/2)]))
    plt.plot(y_plus[:,-1],u_plus[:,-1],label='x={}'.format(x_coord[-1]))
    plt.legend()
    plt.xlabel(r'$y^{+}$')
    plt.ylabel(r'$u^{+}$')

def find_y_value_given_y_plus_mean_value(y_plus_mean=1,nu=1.25e-3,verbose=True,make_plots=False):
    '''
    Calculates the y (y_value) of an horizontal plane y = y_value that has mean(y+) equal 
    to a chosen value (y_plus_mean), being y+ expressed in wall units.
    
    It is returned the mean of y+ because y+(x) depends on x-coordinate, as <U>(x,y) and tau_w(x).
    
    It is optional to print the y_value result and to plot the (fig 1) relation of u+ and y+, 
    depending on the x coordinate, and (fig 2) standarization values delta_v(x) and u_tau(x)
    
    Parameters
    ----------
    y_plus_mean : float         : desired mean value of y+ for the horizontal plane y = y_value
    nu          : float         : fluid viscosity
    verbose     : boolean       : prints result y_value if True
    make_plots  : boolean       : makes plots (fig1, fig2) if True
    
    Returns
    -------
    y_value     : float         : y coordinate of horizontal plane y = y_value so that 
                                  mean(y+(x)) along the plane is equal to y_plus_mean.
    '''
    X_coord, Y_coord, U_Avg = get_averaged_profiles_JHTDB()
    Y_plus, U_plus, U_tau, Delta_v = transform_averaged_profile_to_wall_units(X_coord,Y_coord,U_Avg,nu)
    if make_plots:
        plot_Yplus_Uplus_relation(Y_plus, U_plus, X_coord)
    
    Delta_v_mean = np.mean(Delta_v)
    U_tau_mean = np.mean(U_tau)
    y_value = y_plus_mean * Delta_v_mean
    
    if make_plots:
        _, ax = plt.subplots(2)
        ax[0].plot(X_coord,Delta_v, label=r'$\delta_v(x)$')
        ax[0].plot(X_coord,Delta_v_mean*np.ones(X_coord.shape), '--g', label=r'$\bar{\delta}_v=$'+'{:.3f}'.format(Delta_v_mean))        
        ax[0].set_title(r'Viscous Lengthscale $\delta_V$')
        ax[1].plot(X_coord,U_tau, label=r'$u_{\tau}(x)$')
        ax[1].plot(X_coord,np.mean(U_tau)*np.ones(X_coord.shape),'--g', label=r'$\bar{u}_{\tau}=$'+'{:.3f}'.format(U_tau_mean))
        ax[1].set_title(r'Friction Velocity $u_{\tau}$')
        for i in range(2):
            ax[i].set_xlabel('x')
            ax[i].legend()
        plt.tight_layout()
        plt.show()
    
    if verbose:
        print('\nThe y-plane with mean value y+ = {0} is y = {1:.3f}\n'.format(y_plus_mean, y_value))
        
    return y_value

if __name__ == '__main__':
    y_value = find_y_value_given_y_plus_mean_value(y_plus_mean=1,nu=1.25e-3,verbose=True,make_plots=True)
