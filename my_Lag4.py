import numpy as np


def my_Lag4_1D(x, xn, yn):
    Ax = abs(xn[1]-xn[0])
    nx = int(x/Ax+0.5)
    f = 0
    for i in range(4):
        int_node_x = int(nx - 1 + i)
        lx = 1
        for j in np.linspace(nx-1,nx+2,4).astype('int'):
            if j != int_node_x:
                print(xn[j])
                print(x[int_node_x])
                lx *= (x - xn[j])/(x[int_node_x] - xn[j])
        f += yn[int_node_x]*lx
    return f
        
            
if __name__ == '__main__':
    xn = np.arange(100)
    yn = xn**2-xn+10
    y_int = my_Lag4_1D(5.5,xn,yn)
    print(y_int)
    