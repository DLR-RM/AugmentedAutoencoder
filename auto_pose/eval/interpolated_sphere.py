import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
import os
import time
import numpy as np
from sixd_toolkit.pysixd import pose_error,transform,view_sampler

R_errors = []
# R_gt = transform.random_rotation_matrix()[:3,:3]
# for R in xrange(100000):
#     R_est = transform.random_rotation_matrix()[:3,:3]
#     R_errors.append(pose_error.re(R_est,R_gt))
# plot_R_err_hist2(R_errors,'',bins=90)

azimuth_range = (0, 2 * np.pi)
elev_range = (-0.5 * np.pi, 0.5 * np.pi)
views, _ = view_sampler.sample_views(
    2562, 
    100, 
    azimuth_range, 
    elev_range
)
    
Rs = []
for view in views:
    R_errors.append(pose_error.re(view['R'],views[0]['R']))
    Rs.append(view['R'])

# plot_viewsphere_for_embedding(np.array(Rs),'',np.array(R_errors),save=False)
# plot_R_err_hist2(R_errors,'',bins=45,save=False)


from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

def cart2sph(x, y, z):
    dxy = np.sqrt(x**2 + y**2)
    r = np.sqrt(dxy**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arctan2(z, dxy)
    theta, phi = np.rad2deg([theta, phi])
    return theta % 360, phi, r

def sph2cart(theta, phi, r=1):
    theta, phi = np.deg2rad([theta, phi])
    z = r * np.sin(phi)
    rcosphi = r * np.cos(phi)
    x = rcosphi * np.cos(theta)
    y = rcosphi * np.sin(theta)
    return x, y, z

# random data
# pts = 1 - 2 * np.random.rand(500, 3)
pts = np.array(Rs)[:,2,:].squeeze()

l = np.sqrt(np.sum(pts**2, axis=1))
pts = pts / l[:, np.newaxis]
T = np.array(R_errors)

# naive IDW-like interpolation on regular grid
theta, phi, r = cart2sph(*pts.T)
nrows, ncols = (90,180)
lon, lat = np.meshgrid(np.linspace(0,360,ncols), np.linspace(-90,90,nrows))
xg,yg,zg = sph2cart(lon,lat)
Ti = np.zeros_like(lon)
for r in range(nrows):
    for c in range(ncols):
        v = np.array([xg[r,c], yg[r,c], zg[r,c]])
        angs = np.arccos(np.dot(pts, v))
        idx = np.where(angs == 0)[0]
        if idx.any():
            Ti[r,c] = T[idx[0]]
        else:
            idw = 1 / angs**2 / sum(1 / angs**2)
            Ti[r,c] = np.sum(T * idw)

# set up map projection
map = Basemap(projection='ortho', lat_0=-45, lon_0=10)
# draw lat/lon grid lines every 30 degrees.
map.drawmeridians(np.arange(0, 360, 30))
map.drawparallels(np.arange(-90, 90, 30))
# compute native map projection coordinates of lat/lon grid.
x, y = list(map(lon, lat))
# contour data over the map.
cs = map.contourf(x, y, Ti, 15)
cbar = map.colorbar(cs,location='bottom',pad="5%")
plt.title('Contours of T')
plt.show()