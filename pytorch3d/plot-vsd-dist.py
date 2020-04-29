import csv
import numpy as np
import matplotlib.pyplot as plt

def readCsv(csv_path):
    x = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f, delimiter='\n')
        for line in reader:
            curr_x = float(line[0])
            if(curr_x > 0.1):
                x.append(curr_x)
    return x

def plotDataset(dataset, title=None, save_to=None, y_range=None):
    fig = plt.figure()
    plt.grid(True)

    data = []
    labels = []
    for d in dataset:
        x = readCsv(d[1])
        data.append(x)
        labels.append(d[0] + " total: {0}".format(len(x)))
        #labels.append(d[0])

    plt.hist(data, bins=30, label=labels, alpha=1.0)
    plt.xlabel('VSD')
    plt.ylabel('Occurences')
    plt.legend(loc='upper right')
    axes = plt.gca()
    #axes.set_xlim([-5,x_max])
    #if(y_range is not None):
    #    axes.set_ylim(y_range)
    if(title is not None):
        plt.title(title)
    if(save_to is None):
        plt.show()
    else:
        fig.savefig(save_to, dpi=fig.dpi)

data = []
data.append(('l2-pose-loss','./vsd-pose-1k.csv'))
data.append(('l1-abs-depth-loss','./vsd-depth-1k.csv'))
plotDataset(data,
            title='Object 28 T-LESS - 1k dataset',
            save_to='plot-1k.png',
            y_range=[0.0, 0.05])

data = []
data.append(('l2-pose-loss','./vsd-pose-5k.csv'))
data.append(('l1-abs-depth-loss','./vsd-depth-5k.csv'))
plotDataset(data,
            title='Object 28 T-LESS - 5k dataset',
            save_to='plot-5k.png',
            y_range=[0.0, 0.05])
