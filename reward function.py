import numpy as np
import pandas as pd

df = pd.read_csv('resized_track_coordinates.txt', delimiter=',', header=None, names=['x', 'y'])

rewards = np.zeros((1200,800))

dfx = df['x'].tolist()
dfy = df['y'].tolist()

xypairs = []

for i in range(len(dfx)):
    rewards[(dfx[i])][(dfy[i])] = -10
    xypairs.append(dfx[i],dfy[i])

