# Runge my Kutta

1. RK4_Scatter_PyQt.py: Visualize the evolution of (multiple) dynamical systems, now in 3D* and in PyQt (with [PyQtGraph](https://www.pyqtgraph.org/))!
   * Note: actual 3D is not achievable on a 2D screen; it is assumed that the reader is familiar with the notion of projecting dimensions.
   * Visualization is a bit fiddly in that you might have to change the parameters a bit.
2. RK4toRGB.py: Make trippy-looking images that are very much like the solution to the dynamical system you specified (in a very specific way)! 

PS: [here's a list of ODE](http://www.3d-meier.de/tut19/Seite0.html) (related to chaotic attractors and in german), for your convenience.

![RK4_Scatter_PyQt.py: Aizawa attractor, 10000 points](https://raw.githubusercontent.com/abeRC/runge-my-kutta/main/images/aizawa%2010000%20points.png "RK4_Scatter_PyQt.py: Aizawa attractor, 10000 points")
![RK4toRGB.py: 1 million points of Aizawa attractor goodness, displayed zigzaggedly in a combination of Red, Green and Blue](https://raw.githubusercontent.com/abeRC/runge-my-kutta/main/images/pic_1000000_aiz.png "RK4toRGB.py: 1 million points of Aizawa attractor goodness, displayed zigzaggedly in a combination of Red, Green and Blue")