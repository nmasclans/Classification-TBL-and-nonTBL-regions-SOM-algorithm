import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import axes_grid1


# Get data from h5 file:
def get_instant_data_from_h5_file(file_name):
    file_path = "JHTDB_h5_files/"+file_name
    with h5py.File(file_path, "r") as f:
        coord = f['coord'][:]
        vel = f['vel'][:]
        vel_grad = f['vel_grad'][:]
        time = f.attrs['time']
    return coord, vel, vel_grad, time

def get_instant_data_y_ct_from_h5_file(file_name):
    file_path = "JHTDB_h5_files/" + file_name
    with h5py.File(file_path, "r") as f:
        coord = f['coord'][:]
        vel = f['vel'][:]
        vel_grad = f['vel_grad'][:]
        time = f.attrs['time']
        y_value = f.attrs['y_value']
    return coord, vel, vel_grad, time, y_value

def get_time_averaged_profiles_from_h5_file():
    file_path = "JHTDB_h5_files/Transition_BL_Time_Averaged_Profiles.h5"
    with h5py.File(file_path, "r") as f:
        x = f['x_coor'][:]
        y = f['y_coor'][:]
        um = f['um'][:]
        vm = f['vm'][:]
    return x, y, um, vm

def plot_contour_w_in_xz_plane(coord,vel,time,y_value):
    x = coord[:,:,0]
    z = coord[:,:,2]
    w = vel[:,:,2]
    plt.figure(figsize=(8,3))
    im = plt.contourf(x,z,w,levels=100,linestyle=None)
    add_colorbar(im)
    plt.xlabel('x/L')
    plt.ylabel('z/L')
    plt.title('z-velocity (at y = '+str(y_value)+', time = '+str(time)+')')
    plt.axis('scaled')
    
def plot_contour_u_in_xz_plane(coord,vel,time,y_value):
    x = coord[:,:,0]
    z = coord[:,:,2]
    w = vel[:,:,0]
    plt.figure(figsize=(8,3))
    im = plt.contourf(x,z,w,levels=100,linestyle=None)
    add_colorbar(im)
    plt.xlabel('x/L')
    plt.ylabel('z/L')
    plt.title('x-velocity (at y/L = '+str(y_value)+', time = '+str(time)+')')
    plt.axis('scaled')
    
def plot_contour_dudy_in_xz_plane(coord,vel_grad,time,y_value):
    x = coord[:,:,0]
    z = coord[:,:,2]
    dudy = vel_grad[:,:,1]
    plt.figure(figsize=(8,3))
    im = plt.contourf(x,z,dudy,levels=100,linestyle=None)
    add_colorbar(im)
    plt.xlabel('x/L')
    plt.ylabel('z/L')
    plt.title('du/dy velocity gradient (at y = '+str(y_value)+', time = '+str(time)+')')
    plt.axis('scaled')
    
def plot_contour_dwdy_in_xz_plane(coord,vel_grad,time,y_value):
    x = coord[:,:,0]
    z = coord[:,:,2]
    dwdy = vel_grad[:,:,7]
    plt.figure(figsize=(8,3))
    im = plt.contourf(x,z,dwdy,levels=100,linestyle=None)
    add_colorbar(im)
    plt.xlabel('x/L')
    plt.ylabel('z/L')
    plt.title('dw/dy velocity gradient (at y = '+str(y_value)+', time = '+str(time)+')')
    plt.axis('scaled')

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

if __name__ == '__main__':
    
    #Coord, Vel, Vel_gradient, Time = get_instant_data_from_h5_file("JHTDB_time_0-0_n_25x10x15.h5")
    #X_Avg_Grid, Y_Avg_Grid, Um_Avg_Grid, Vm_Avg_Grid = get_time_averaged_profiles_from_h5_file()
    
    # Data at constant y/L = 0.5
    Coord, Vel, Vel_Grad, Time, Y_Value = get_instant_data_y_ct_from_h5_file("JHTDB_time_10-0_n_80x24_y_0-5.h5")
    plot_contour_w_in_xz_plane(Coord,Vel,Time,Y_Value)
    plot_contour_u_in_xz_plane(Coord,Vel,Time,Y_Value)
    
    # Data at constant y/L = 0
    Coord, Vel, Vel_Grad, Time, Y_Value = get_instant_data_y_ct_from_h5_file("JHTDB_time_10-0_n_80x24_y_0-0.h5")
    plot_contour_dudy_in_xz_plane(Coord,Vel_Grad,Time,Y_Value)
    plot_contour_dwdy_in_xz_plane(Coord,Vel_Grad,Time,Y_Value)
    
    plt.show()
