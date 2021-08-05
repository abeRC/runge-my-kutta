#!/usr/bin/env python3

import numpy as np
import sys
from numba import jit # just-in-time compiler
from PIL import Image

############# Attractor equations and default initial conditions: #############
t0 = 0
#"""Aizawa attractor
x0, y0, z0 = 0.1, 0, 0
a, b, c, d, e, f = 0.95, 0.7, 0.6, 3.5, 0.25, 0.1
@jit
def aizawa (t, v):
    x = v[0]
    y = v[1]
    z = v[2]

    return np.array([(z-b)*x -d*y, 
        d*x + (z-b)*y, 
        c + a*z - z**3/3 - (x**2 + y**2)*(1 + e*z) + f*z*x**3])
FUNC = aizawa
#"""

"""Chen-Lee attractor
x0, y0, z0 = 1, 0, 4.5
a, b, c = 5, -10, -0.38
@jit
def chen_lee (t, v):
    x = v[0]
    y = v[1]
    z = v[2]
    
    return np.array([a*x - y*z,
                     b*y+x*z,
                     c*z+x*y/3])
FUNC = chen_lee
"""

"""Newton-Leipnik attractor
x0, y0, z0 = 0.349, 0, -0.16
a, b = 0.4, 0.175
@jit
def newton_leipnik (t, v):
    x = v[0]
    y = v[1]
    z = v[2]
    
    return np.array([-a*x+y+10*y*z,
                     -x-0.4*y+5*x*z,
                     b*z-5*x*y])
FUNC = newton_leipnik
"""

u0 = np.array([x0, y0, z0]) # group default initial condition into np array

@jit
def runge_kutta (F, u0, N):
    """Given a vector-valued function F; an initial vector u0; initial time;
    and number of points; 
    calculates an approximate solution to u'[j] = fs[j](t, u) 
    using the fourth-order runge-kutta method.
    Returns a matrix of (N+1) points approximating the solution to the ODE."""
    
    no_eq = len(u0) # number of dimensions

    # reusable variables
    t = t0
    u = u0.copy()
    k1, k2, k3, k4 = [np.zeros(no_eq) for _ in range(4)] # 4 empty arrays of size no_eq

    points = np.zeros((no_eq, N+1))
    points[:,0] = u0

    # runge-kutta step
    for i in range(1, N+1):
        k1 = h*F(t, u)
        k2 = h*F(t + h/2, u + 0.5*k1)
        k3 = h*F(t + h/2, u + 0.5*k2)
        k4 = h*F(t + h, u + k3)

        # update u, t and add the new point
        u += (k1 + 2*k2 + 2*k3 + k4)/6
        t = t0 + i*h
        points[:,i] = u

    return points

def create_image (point_list):
    # unpack points
    x, y, z = point_list
    minx, maxx = np.amin(x), np.amax(x)
    miny, maxy = np.amin(y), np.amax(y)
    minz, maxz = np.amin(z), np.amax(z)
    
    # preprocess x,y,z
    preparex = lambda v: (v+abs(minx)) * 255/(maxx+abs(minx))
    preparey = lambda v: (v+abs(miny)) * 255/(maxy+abs(miny))
    preparez = lambda v: (v+abs(minz)) * 255/(maxz+abs(minz))
    x = preparex(x).astype(int)
    y = preparey(y).astype(int)
    z = preparez(z).astype(int)
    
    # create empty image
    no_points = point_list.shape[1]
    
    N = int(np.sqrt(no_points))
    im = Image.new('RGB', (N, N))
    ld = im.load()
    
    # fill in pixels using a zig-zag pattern
    cnt = 0
    for j in range(N):
        r = range(0, j+1)
        for i in r if (j % 2 == 1) else reversed(r):
            r, g, b = x[cnt], y[cnt], z[cnt]
            ld[i, j-i] = (r, g, b)
            cnt += 1
    for j in range(N, 2*N-1):
        r = range(j-N+1, N)
        for i in r if (j % 2 == 1) else reversed(r):
            r, g, b = x[cnt], y[cnt], z[cnt]
            ld[i, j-i] = (r, g, b)
            cnt += 1
    print(f"Saving image as pic_{FUNC.__name__}_{sys.argv[1]}_{h}.png ...")
    im.save(f"pic_{FUNC.__name__}_{sys.argv[1]}_{h}.png")

def usage ():
    print("""Usage: python RK4toRGB.py number_of_points [step_size]
    step_size is an optional float and should be less than 1.""")

def main ():
    global h
    
    # parse arguments
    help = False
    for arg in sys.argv:
        if arg == "-h" or "help" in arg:
            help = True
    if help:
        usage()
        sys.exit(0)
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("Error: wrong number of arguments.\n")
        usage()
        sys.exit(1)
    N = int(sys.argv[1])
    if len(sys.argv) == 3:
        h = float(sys.argv[2])
    else:
        tf = 100 # default value
        h = (tf - t0)/N
    
    print("N =", N, ", h =", h)

    points = runge_kutta(FUNC, u0, N)
    create_image(points)

if __name__ == "__main__" :
    main()
