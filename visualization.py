import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D


def visualizeReebNodes(streamlines, H, node_loc, colorT = '#08c29d', colorN = 'r', save = False):
    plt.figure(figsize = (20,20))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x_data = []
    y_data = []
    z_data = []
#     print(H.nodes)

    for key in H.nodes():
#         key = str(key)
#         print(node_loc[key][0], node_loc[key][1],node_loc[key][2])
        x_data.append(node_loc[key][0])
        y_data.append(node_loc[key][1])
        z_data.append(node_loc[key][2])
    for i in range(len(streamlines)):
        xdata = []
        ydata = []
        zdata = []
        for j in streamlines[i]:
            xdata.append(j[0])
            ydata.append(j[1])
            zdata.append(j[2])
        ax.plot3D(xdata,ydata,zdata,color=colorT, label = "Trajectory"+str(i+1), lw = 1);
    ax.scatter(x_data,y_data,z_data, color = colorN, lw = 4);
    ax.set_xlabel('$X$', fontsize = 20)
    ax.set_ylabel('$Y$', fontsize = 20)
    ax.set_zlabel('$Z$', fontsize = 20)
#     ax.set_title(trkpathI)
    plt.axis("off")
    custom_lines = [Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="red")]
    ax.legend(custom_lines, [len(H.nodes)],fontsize = 10)
#     ax.set_title("test")
    if save:
        plt.savefig("Reeb"+str(trkpathI)+".svg", transparent=True)    

def visualizeStreams(streamlines, color = '#24a0a7',  save = False):
    plt.figure(figsize = (8,6))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in range(len(streamlines)):
        xdata = []
        ydata = []
        zdata = []
        for j in streamlines[i]:
            xdata.append(j[0])
            ydata.append(j[1])
            zdata.append(j[2])
        ax.plot3D(xdata,ydata,zdata,color= color, label = "Trajectory"+str(i+1), lw = 4);
