import matplotlib.pyplot as plt
import PIL
import scipy
from matplotlib import image as img
import cv2
import pandas as pd
from scipy.cluster.vq import whiten
from scipy.cluster.vq import kmeans
import numpy as np

list_of_colors=[[255,255,0],[255,0,0],[0,0,255],[255,255,255],[0,0,0],[135, 150, 181]]

def sorted_cluster_centers_(cluster_centers):
    magnitude = []
    for center in cluster_centers:
        magnitude.append(np.sqrt(center.dot(center)))
    idx_argsort = np.argsort(magnitude)
    cluster_centers = cluster_centers[idx_argsort]
    return cluster_centers

def closest_color(list_of_colors, color):
    colors = np.array(list_of_colors)
    color = np.array(color)
    distances = np.sqrt(np.sum((colors-color)**2,axis=1))
    index_of_shortest = np.where(distances==np.amin(distances))
    shortest_distance = colors[index_of_shortest]
    return shortest_distance

def getClusters(img):
    image = cv2.imread(img)
    image = cv2.resize(image,(200,200))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r = []
    g = []
    b = []
    for line in image:
        for pixel in line:
            temp_r, temp_g, temp_b = pixel
            r.append(temp_r)
            g.append(temp_g)
            b.append(temp_b)
            
    df = pd.DataFrame({'red': r,'blue': b,'green': g})
    df['scaled_red'] = df['red'].astype('float64')
    df['scaled_blue'] = df['blue'].astype('float64')
    df['scaled_green'] = df['green'].astype('float64')
    cluster_centers, distortion = kmeans(df[['scaled_red', 'scaled_green', 'scaled_blue']], 2)
    cluster_centers=sorted_cluster_centers_(cluster_centers)
    for cluster in cluster_centers:
        if((int(cluster[0]) in range(30,120)) and (int(cluster[1]) in range(30,173)) and (int(cluster[2]) in range(0,140))):
            pass
        else:
            return cluster

cluster=getClusters('../image5.JPG')
print(cluster)
assigned_color = closest_color(list_of_colors, cluster)[0]
assigned_color = (int(assigned_color[0]), int(assigned_color[1]), int(assigned_color[2]))
print(assigned_color)