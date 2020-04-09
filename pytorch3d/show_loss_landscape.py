import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import sys
import os
from mpl_toolkits.mplot3d import Axes3D

def voronoi_plot(points_in, losses):
    from matplotlib import colors
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.pyplot as plt
    from scipy.spatial import SphericalVoronoi
    from mpl_toolkits.mplot3d import proj3d
    # get input points in correct format
    cart = [point['cartesian'] for point in points_in]
    points = np.array(cart)
    center = np.array([0, 0, 0])
    radius = 1
    # calculate spherical Voronoi diagram
    sv = SphericalVoronoi(points, radius, center)
    # sort vertices (optional, helpful for plotting)
    sv.sort_vertices_of_regions()
    # generate plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot the unit sphere for reference (optional)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='y', alpha=0.1)

    # normalize and map losses to colormap
    mi = min(losses)
    ma = max(losses)

    norm = matplotlib.colors.Normalize(vmin=mi, vmax=ma, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)

    loss_color = [mapper.to_rgba(l) for l in losses]

    # indicate Voronoi regions (as Euclidean polygons)
    for i in range(len(sv.regions)):
        region = sv.regions[i]
        random_color = colors.rgb2hex(loss_color[i])
        polygon = Poly3DCollection([sv.vertices[region]], alpha=1.0)
        polygon.set_color(random_color)
        ax.add_collection3d(polygon)
    plt.show()



def plot_points(points, losses):
    cart = [point['cartesian'] for point in points]
    x, y, z = zip(*cart)
    #fig = plt.figure()

    ma = max(losses)
    mi = min(losses)

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    print(z)

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=losses, s=700)
    plt.show()

    #norm = matplotlib.colors.Normalize(vmin=mi, vmax=ma, clip=True)
    #mapper = cm.ScalarMappable(norm=norm, cmap=cm.Greys_r)

    #loss_color = [mapper.to_rgba(l) for l in losses]


    #norm = matplotlib.colors.Normalize(mi, ma)
    #m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
    #m.set_array([])
    #fcolors = m.to_rgba(losses)

    #X, Y = np.meshgrid(x, y)    # 50x50
    #Z = np.outer(z.T, z)        # 50x50

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.plot_trisurf(x, y, z, rstride=1, cstride=1, color=losses, shade=0, cmap=cm.Greys_r)
    #ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=fcolors, shade=0)

def main():
    path = sys.argv[1]
    points = np.load(os.path.join(path, 'points.npy'))
    losses = np.load(os.path.join(path, 'losses.npy'))
    voronoi_plot(points, losses)


if __name__ == '__main__':
    main()
