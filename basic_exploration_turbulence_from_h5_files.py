import h5py
import matplotlib.pyplot as plt
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
        wm = f['wm'][:]
    return x, y, um, vm, wm

def plot_contour_u_in_xz_plane(coord,vel,time,y_value,file_name):
    x = coord[:,:,0]
    z = coord[:,:,2]
    u = vel[:,:,0]
    plt.figure(figsize=(8,3))
    im = plt.contourf(x,z,u,levels=100,linestyles=None)
    add_colorbar(im)
    plt.xlabel('x/L')
    plt.ylabel('z/L')
    plt.title('x-velocity (at y/L = '+str(y_value)+', time = '+str(time)+')')
    plt.axis('scaled')
    file_path = "JHTDB_basic_exploration_figures/" + file_name[:-3] + '_contour_u_plane_xz.png'
    plt.savefig(file_path)

def plot_contour_w_in_xz_plane(coord,vel,time,y_value,file_name):
    x = coord[:,:,0]
    z = coord[:,:,2]
    w = vel[:,:,2]
    plt.figure(figsize=(8,3))
    im = plt.contourf(x,z,w,levels=100,linestyles=None)
    add_colorbar(im)
    plt.xlabel('x/L')
    plt.ylabel('z/L')
    plt.title('z-velocity (at y/ = '+str(y_value)+', time = '+str(time)+')')
    plt.axis('scaled')
    file_path = "JHTDB_basic_exploration_figures/" + file_name[:-3] + '_contour_w_plane_xz.png'
    plt.savefig(file_path)
    
def plot_contour_dudy_in_xz_plane(coord,vel_grad,time,y_value,file_name):
    x = coord[:,:,0]
    z = coord[:,:,2]
    dudy = vel_grad[:,:,1]
    plt.figure(figsize=(8,3))
    im = plt.contourf(x,z,dudy,levels=100,linestyles=None)
    add_colorbar(im)
    plt.xlabel('x/L')
    plt.ylabel('z/L')
    plt.title('du/dy velocity gradient (at y/L = '+str(y_value)+', time = '+str(time)+')')
    plt.axis('scaled')
    file_path = "JHTDB_basic_exploration_figures/" + file_name[:-3] + '_contour_dudy_plane_xz.png'
    plt.savefig(file_path)
    
def plot_contour_dwdy_in_xz_plane(coord,vel_grad,time,y_value,file_name):
    x = coord[:,:,0]
    z = coord[:,:,2]
    dwdy = vel_grad[:,:,7]
    plt.figure(figsize=(8,3))
    im = plt.contourf(x,z,dwdy,levels=100,linestyles=None)
    add_colorbar(im)
    plt.xlabel('x/L')
    plt.ylabel('z/L')
    plt.title('dw/dy velocity gradient (at y/L = '+str(y_value)+', time = '+str(time)+')')
    plt.axis('scaled')
    file_path = "JHTDB_basic_exploration_figures/" + file_name[:-3] + '_contour_dwdy_plane_xz.png'
    plt.savefig(file_path)

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
    
    Coord, Vel, Vel_gradient, Time = get_instant_data_from_h5_file("JHTDB_time_0-0_n_25x10x15.h5")
    X_Avg, Y_Avg, Um, Vm, Wm = get_time_averaged_profiles_from_h5_file()
    
    # Data at constant y/L = 0.5
    File_Name = "JHTDB_time_10-0_n_831x512_y_0-5.h5"
    Coord, Vel, Vel_Grad, Time, Y_Value = get_instant_data_y_ct_from_h5_file(File_Name)
    plot_contour_w_in_xz_plane(Coord,Vel,Time,Y_Value,File_Name)
    plot_contour_u_in_xz_plane(Coord,Vel,Time,Y_Value,File_Name)
    
    # Data at constant y/L = 0
    File_Name = "JHTDB_time_10-0_n_831x512_y_0-0.h5"
    Coord, Vel, Vel_Grad, Time, Y_Value = get_instant_data_y_ct_from_h5_file(File_Name)
    plot_contour_dudy_in_xz_plane(Coord,Vel_Grad,Time,Y_Value,File_Name)
    plot_contour_dwdy_in_xz_plane(Coord,Vel_Grad,Time,Y_Value,File_Name)
    
    # Data at constant y/L = 0.03, which correspons to a plane with mean y+ = 1
    File_Name = "JHTDB_time_10-0_n_831x512_y_0-03.h5"
    Coord, Vel, Vel_Grad, Time, Y_Value = get_instant_data_y_ct_from_h5_file(File_Name)
    plot_contour_w_in_xz_plane(Coord,Vel,Time,Y_Value,File_Name)
    plot_contour_u_in_xz_plane(Coord,Vel,Time,Y_Value,File_Name)
    plot_contour_dudy_in_xz_plane(Coord,Vel_Grad,Time,Y_Value,File_Name)
    plot_contour_dwdy_in_xz_plane(Coord,Vel_Grad,Time,Y_Value,File_Name)
    
    plt.show()
