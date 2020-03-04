import csv
import numpy as np
import matplotlib.pyplot as plt

csv_name = "../output/depth/exp-normalized/train-loss.csv"

x = []
with open(csv_name, 'r') as f:
    reader = csv.reader(f, delimiter='\n')
    for line in reader:
        x.append(float(line[0]))

#x = x[:60]
plt.grid(True)
plt.plot(x)
#plt.scatter(np.arange(len(x)),x)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()
