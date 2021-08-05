 #!/usr/bin/env python3

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import sys

h = 0.1 # step size

############# Attractor equations and default initial conditions: #############
t0 = 0
EXTRA_DISTANCE = 0
EXTRA_SIZE = 0

"""Aizawa attractor
x0, y0, z0 = 0.1, 0.1, 0.1
a, b, c, d, e, f = 0.95, 0.7, 0.6, 3.5, 0.25, 0.1
def aizawa (t, v):
    x = v[0]
    y = v[1]
    z = v[2]

    return np.array([(z-b)*x -d*y, 
        d*x + (z-b)*y, 
        c + a*z - z**3/3 - (x**2 + y**2)*(1 + e*z) + f*z*x**3])
FUNC = aizawa
"""

"""Chen-Lee attractor
x0, y0, z0 = 1, 0, 4.5
a, b, c = 5, -10, -0.38
EXTRA_DISTANCE = 40
def chen_lee (t, v):
    x = v[0]
    y = v[1]
    z = v[2]
    
    return np.array([a*x - y*z,
                     b*y+x*z,
                     c*z+x*y/3])
FUNC = chen_lee
"""

#"""Newton-Leipnik attractor
x0, y0, z0 = 0.349, 0, -0.16
a, b = 0.4, 0.175
EXTRA_DISTANCE, EXTRA_SIZE = -3, -0.05
def newton_leipnik (t, v):
    x = v[0]
    y = v[1]
    z = v[2]
    
    return np.array([-a*x+y+10*y*z,
                     -x-0.4*y+5*x*z,
                     b*z-5*x*y])
FUNC = newton_leipnik
#"""

u0_good = np.array([x0, y0, z0]) # group default initial condition into np array
    
class ScatterBalls:
    def __init__ (self, F, u0, t0):
        global scatters

        self.t = t0
        self.u = u0.copy()
        self.dims = len(u0) # usually 3
        self.k1, self.k2, self.k3, self.k4 = [np.zeros(self.dims) for _ in range(4)] # 4 empty arrays of size no_eq
        
        self.points = np.zeros((self.dims, N+1))
        self.points[:,0] = u0
        self.F = F
        
        # execute runge-kutta and setup PyQtGraph plot
        self.runge_kutta_initial(self.F, t0, N)
        self.setup_plot()

        # add to list of ScatterBalls
        scatters.append(self)
        print("New scatter with u0 = ", u0)

    def runge_kutta_step (self, F):
        """Perform iteration step for the runge-kutta method by updating u and k1,k2,k3,k4."""
        
        self.k1 = h*F(self.t, self.u)
        self.k2 = h*F(self.t + h/2, self.u + 0.5*self.k1)
        self.k3 = h*F(self.t + h/2, self.u + 0.5*self.k2)
        self.k4 = h*F(self.t + h, self.u + self.k3)

        self.u += (self.k1 + 2*self.k2 + 2*self.k3 + self.k4)/6 # update u
        
    def runge_kutta_initial (self, F, t0, N):
        """Given a vector-valued function F; an initial vector u0; initial time;
        and number of points; 
        calculates an approximate solution to u'[j] = fs[j](t, u) 
        using the fourth-order runge-kutta method.
        Returns a matrix of (N+1) points approximating the solution to the ODE."""
        
        global h

        for i in range(1, N+1):
            self.runge_kutta_step(F) # perform runge-kutta step
            
            # update time and array of points
            self.t = t0 + i*h
            self.points[:,i] = self.u

    def setup_plot (self):
        global w, N

        pos = self.points.T # points should be rows instead of columns
        rng = np.random.default_rng()
        size = (0.06+EXTRA_SIZE)*rng.random(size=N+1) # randomize size for better effect
        color = (rng.random(), rng.random(), rng.random(), 1)
        
        # create scatter plot and add to window
        scatter = gl.GLScatterPlotItem(pos=pos, size=size, color=color, pxMode=False)
        self.s = scatter
        w.addItem(scatter)
    
    def update_data (self):
        global h

        # execute step
        self.runge_kutta_step(self.F)

        # increase time, remove old point, add a new point and update scatter
        self.t += h
        self.points = np.column_stack([self.points[:,1:], self.u.copy()])
        self.s.setData(pos=self.points.T)

def make_grid_cube (w : gl.GLViewWidget):
    """Create a half-cube of grid rectangles"""
    
    gx = gl.GLGridItem()
    gx.rotate(90, 0, 1, 0)
    gx.translate(-10, 0, 0)
    w.addItem(gx)
    gy = gl.GLGridItem()
    gy.rotate(90, 1, 0, 0)
    gy.translate(0, -10, 0)
    w.addItem(gy)
    gz = gl.GLGridItem()
    gz.translate(0, 0, -10)
    w.addItem(gz)

class MyGLViewWidget (gl.GLViewWidget):
    def __init__(self, *args, **kwargs):
        gl.GLViewWidget.__init__(self, *args, **kwargs) # call parent constructor
    
    # override (empty) event handler for mouse releases with our own
    def mouseReleaseEvent (self, e):
        global t0, u0_good

        rng = np.random.default_rng()
        x0, y0, z0 = u0_good
        # hopefully not a bad choice for u0
        u0 = np.array([(rng.random())*x0, (rng.random())*y0, (rng.random())*z0])
        s = ScatterBalls(FUNC, u0, t0)

def setup_graphics ():
    """Setup PyQtGraph 3D graphics."""
    global w

    # setup graphics widget
    w = MyGLViewWidget()
    w.opts['distance'] = 5 + EXTRA_DISTANCE # increase to, for instance, 50 if view is too zoomed in
    make_grid_cube(w)

    # display window
    w.setWindowTitle('RK4_Scatter_PyQt')
    w.show()

def print_help ():
    print("""Usage: python RK4_Scatter_PyQt.py [N]
        N: number of points for dynamic plot; defaults to 1000 if not specified.""")

def main ():
    global N, scatters

    # parse arguments
    help = False
    for arg in sys.argv:
        if arg == "-h" or "help" in arg:
            help = True
    if help:
        print_help()
        sys.exit(0)
    if len(sys.argv) == 1:
        print("Number of points not specified; defaulting to 1000...")
        N = 1000
    elif len(sys.argv) == 2:
        N = int(sys.argv[1]) # number of points for each dynamic plot
    else:
        print("Error: this program takes either 0 or 1 argument.\n")
        print_help()
        sys.exit(1)
    
    # setup Qt app and create objects
    app = pg.mkQApp()
    setup_graphics()
    scatters = [] # s will add itself to the list
    s = ScatterBalls(FUNC, u0_good, t0)

    # setup update function
    time = QtCore.QTimer()
    time.timeout.connect(update)
    time.start(50)

    # start the app
    print("\nClick the screen to spawn more scatter plots!")
    app.exec_()
    

def update ():
    global scatters

    for s in scatters:
        s.update_data()

if __name__ == "__main__":
    main()
