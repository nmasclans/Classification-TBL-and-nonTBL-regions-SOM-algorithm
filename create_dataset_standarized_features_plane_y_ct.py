import h5py
import numpy as np
from scipy.interpolate import griddata


def interpolate_averaged_velocity_in_instant_data_y_ct_grid(coord, y_value):
    '''
    Interpolates averaged velocity data (from "JHTDB_h5_files/Transition_BL_Time_Averaged_Profiles.h5" file, 
    using function "get_averaged_profiles_JHTDB") to the simulation snapshot grid of an horizontal plane (Y_Value ct.)
    
    Parameters
    ----------
    coord: np.ndarray(Nx, Nz, dim=3)    : simulation grid (from time snapshot) where to interpolate the avg. velocities
    y_value : float                     : y-coordinate of the simulation plane (ct. along the plane)
    
    Values 
    ------
    vel_avg : np.ndarray(Nx, Nz, 2)     : velocity averages interpolated in the simulation instant grid in horitz plane y = Y_Value
                                          [:,:,0] is the U_Avg, averaged velocity in x-direction
                                          [:,:,1] is the V_avg, averaged velocity in y-direction
                                          ** W_Avg (avg. vel. in z-direction) not calculated, in boundary layer is ~0 !
                                          ** then, W_Avg not used by the clustering algorithm:) 
     
    '''
    # Grid where velocity averaged is known: (xm, ym)
    x, y, um, vm, _ = get_averaged_profiles_JHTDB()
    xm, ym = np.meshgrid(x,y)
    
    # Grid where to interpolated the averaged velocity == instant data y-ct grid, from simulation snapshood
    xm_interp = coord[:,0,0] 
    ym_interp = Y_Value*np.ones(xm_interp.shape)
    um_interp = griddata(points = (xm.reshape(-1),ym.reshape(-1)),
                        values = um.reshape(-1),
                        xi = (xm_interp,ym_interp), method='linear')
    vm_interp = griddata(points = (xm.reshape(-1),ym.reshape(-1)),
                        values = vm.reshape(-1),
                        xi = (xm_interp,ym_interp), method='linear')
    # Extrapolate result to z-coord: um/vm/wm are equal along z-coord -> np.newaxis in z index
    vel_avg = np.empty(Coord.shape)
    vel_avg[:,:,0] = um_interp[:,np.newaxis]  
    vel_avg[:,:,1] = vm_interp[:,np.newaxis]
    return vel_avg

def get_averaged_profiles_JHTDB():
    '''
    Returns
    -------
    x : numpy.ndarray(Nx,) -- (3320,)
        x-coordinates of the grid where averaged profiles are given
    y : numpy.ndarray(Ny,) -- (224,)
        y-coordinates of the grid where averaged profiles are given
    um : numpy.ndarray(Ny, Nx) -- (224, 3320)
        x-velocity  averaged on time and on z-coordinate (<U>(x,y))
    vm : numpy.ndarray(Ny, Nx) -- (224, 3320)
        y-velocity averaged on time and on z-coordinate  (<V>(x,y))
    wm : numpy.ndarray(Ny, Nx) -- (224, 3320)
        z-velocity  averaged on time and on z-coordinate (<W>(x,y))
    '''
    file_path = "JHTDB_h5_files/Transition_BL_Time_Averaged_Profiles.h5"
    with h5py.File(file_path, "r") as f:
        x = f['x_coor'][:]
        y = f['y_coor'][:]
        um = f['um'][:] # <U>(x,y), averaged on time and on z-coordinate (as z-coord has no statistic influence in boundary layer, Pope ch. 7.3)
        vm = f['vm'][:]
        wm = f['wm'][:]
    return x, y, um, vm, wm

def get_instant_data_y_ct_from_h5_file(file_name):
    '''
    Imports y-plane simulation data from a file (.h5)
    
    Parameters:
    ----------
    file_name : str         : file name of simulation data on a plane, i.e. 'JHTDB_time_10-0_n_831x512_y_0-03.h5'
                              ** the file must be stored inside 'JHTDB_h5_files' folder
    
    Values:
    -------
    coord     : ndarray(Nx,Nz,3)  : x,y,z coordinates  (index #2), along simulation plane grid (Nx x Nz) (index #0, #1)
    vel       : ndarray(Nx,Nz,3)  : u,v,w velocities   (index #2), along simulation plane grid (Nx x Nz)
    vel_fluct : ndarray(Nx,Nz,2)  : uf,vf fluctuations (index #2), along simulation plane grid (Nx x Nz)
    vel_grad  : ndarray(Nx,Nz,9)  : velocity gradients (index #2), along simulation plane grid (Nx x Nz)
                                    index #2:   [0] du/dx, [1] du/dy, [2] du/dz
                                                [3] dv/dx, [4] dv/dy, [5] dv/dz
                                                [6] dw/dx, [7] dw/dy, [8] dw/dz
    time     : float             : simulation time
    y_value  : float             : y-coord value of the simulation plane (horizontal plane, y ct.)
    
    '''
    file_path = "JHTDB_h5_files/" + file_name
    with h5py.File(file_path, "r") as f:
        coord = f['coord'][:]
        vel = f['vel'][:]
        vel_grad = f['vel_grad'][:]
        time = f.attrs['time']
        y_value = f.attrs['y_value']
    return coord, vel, vel_grad, time, y_value

def save_standarized_features_dataset(coord, vel, vel_fluct, vel_grad, file_name):
    '''
    Standarizes the fields along the horizontal plane, and saves the dataset in a file
    
    For horizontal plane (y ct.), it is constructed a dataset of 16 standarized features:
        (1-3) instantaneous velocity: u_s,  v_s, w_s
        (4-5) velocity fluctuations:  uf_s, vf_s
        (6-14) velocity gradients:    dudx_s, dudy_s, dudz_, dvdx_s, dvdy_s, dvdz_s, dwdx_s, dwdy_s, dwdz_s
        (15-16) x-coordinates:           x_s, z_s
        
    The input fields are in grid shape (Nx, Nz, ...), but the saved datasets are all in columns (Nx*Nz)
    
    Parameters
    ---------
    coord     : ndarray(Nx,Nz,3)  : x,y,z coordinates  (index #2), along simulation plane grid (Nx x Nz) (index #0, #1)
    vel       : ndarray(Nx,Nz,3)  : u,v,w velocities   (index #2), along simulation plane grid (Nx x Nz)
    vel_fluct : ndarray(Nx,Nz,2)  : uf,vf fluctuations (index #2), along simulation plane grid (Nx x Nz)
    vel_grad  : ndarray(Nx,Nz,9)  : velocity gradients (index #2), along simulation plane grid (Nx x Nz)
                                    index #2:   [0] du/dx, [1] du/dy, [2] du/dz
                                                [3] dv/dx, [4] dv/dy, [5] dv/dz
                                                [6] dw/dx, [7] dw/dy, [8] dw/dz
    file_name : str               : file name (including .h5) of the simulation instant dataset in plane grid
                                    i.e. "JHTDB_time_10-0_n_831x512_y_0-03.h5"
    
    Output
    ------
    The standarized dataset .h5 will be created in the file: JHTDB_standarized_features/'file_name'_standarized_features.h5,
    i.e. JHTDB_standarized_features/JHTDB_time_10-0_n_831x512_y_0-03_standarized_features.h5
    
    '''
    u_s = standarize_field(vel[:,:,0])
    v_s = standarize_field(vel[:,:,1])
    w_s = standarize_field(vel[:,:,2])
    uf_s = standarize_field(vel_fluct[:,:,0])
    vf_s = standarize_field(vel_fluct[:,:,1])
    dudx_s = standarize_field(vel_grad[:,:,0])
    dudy_s = standarize_field(vel_grad[:,:,1])
    dudz_s = standarize_field(vel_grad[:,:,2])
    dvdx_s = standarize_field(vel_grad[:,:,3])
    dvdy_s = standarize_field(vel_grad[:,:,4])
    dvdz_s = standarize_field(vel_grad[:,:,5])
    dwdx_s = standarize_field(vel_grad[:,:,6])
    dwdy_s = standarize_field(vel_grad[:,:,7])
    dwdz_s = standarize_field(vel_grad[:,:,8])
    x_s = standarize_field(coord[:,:,0])
    z_s = standarize_field(coord[:,:,2])
    
    # Save standarized features dataset in h5 file:
    file_name = file_name[:-3] + '_standarized_features.h5'
    file_path = "JHTDB_standarized_features/" + file_name
    with h5py.File(file_path, "w") as f:
        f.create_dataset('u_s',data=u_s)
        f.create_dataset('v_s',data=v_s)
        f.create_dataset('w_s',data=w_s)
        f.create_dataset('uf_s',data=uf_s)
        f.create_dataset('vf_s',data=vf_s)
        f.create_dataset('dudx_s',data=dudx_s)
        f.create_dataset('dudy_s',data=dudy_s)
        f.create_dataset('dudz_s',data=dudz_s)
        f.create_dataset('dvdx_s',data=dvdx_s)
        f.create_dataset('dvdy_s',data=dvdy_s)
        f.create_dataset('dvdz_s',data=dvdz_s)
        f.create_dataset('dwdx_s',data=dwdx_s)
        f.create_dataset('dwdy_s',data=dwdy_s)
        f.create_dataset('dwdz_s',data=dwdz_s)
        f.create_dataset('x_s',data=x_s)
        f.create_dataset('z_s',data=z_s)
    print("\nDatabase file created: " + file_path) 

def standarize_field(f):
    ''''
    Standarize field (mean = 0, std = 1) described in the instant simulation grid (Nx, Nz)
    
    Parameters
    ----------
    f : np.ndarray(Nx, Nz)  : field (coordinates, velocity, velocity gradients,etc.) described in simulation instant coord
    
    Values 
    ------
    f_s : np.ndarray(Nx*Nz) : standarized field along the simulation instant grid, reshaped(-1) as a column
    
    '''
    f = f.reshape(-1)
    return (f-np.mean(f))/np.std(f)
    

if __name__ == '__main__':
    
    File_Name = "JHTDB_time_10-0_n_831x512_y_0-03.h5"
    Coord, Vel, Vel_Grad, Time, Y_Value = get_instant_data_y_ct_from_h5_file(File_Name)
    Vel_Avg = interpolate_averaged_velocity_in_instant_data_y_ct_grid(Coord, Y_Value)
    Vel_Fluct = Vel_Avg - Vel
    save_standarized_features_dataset(Coord, Vel, Vel_Fluct, Vel_Grad, File_Name)
