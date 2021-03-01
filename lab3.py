import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np 
from PIL import Image 
from scipy import ndimage
import skimage
from skimage.io import imread, imshow
import skimage.feature
from skimage.feature import peak_local_max, corner_peaks, corner_shi_tomasi, corner_harris
from math import sqrt 
import json 
import pandas 
import yaml


def binary_img(img,threshold=1):
    img[img > threshold] = True
    img[img < -threshold] = True
    img[img != True] = False
    return img

def edge_detection(img,k):
    img = ndimage.convolve(img,k,mode="constant",cval=0.0)
    img = binary_img(img)
    return img

def show_img_k(img,k,show=True):
    fig,axes = plt.subplots(1,2,figsize=(15,6))
    axes[0].set_title("Kernel")
    axes[0].imshow(k)
    axes[1].set_title("Convolved image")
    axes[1].imshow(img)
    if show:
        plt.show()

def plot_corners(img,corners):
    x,y,_ = zip(*corners)
    fig,axes = plt.subplots(1,2,figsize=(15,6))
    axes[0].set_title("Corners")
    axes[0].scatter(x,y)
    axes[0].set_ylim(axes[0].get_ylim()[::-1])
    axes[0].xaxis.tick_top()                     # and move the X-Axis      
    axes[0].yaxis.tick_left()
    axes[1].set_title("image")
    axes[1].imshow(img)
    plt.show()

def points_match(a,b,distance=20):
    x = (b[0]-a[0]) ** 2
    y = (b[1]-a[1]) ** 2
    euclidian = sqrt(x+y)
    if euclidian<=distance:
        return True, euclidian 
    return False, 0 

def matching_points(A,B,dist=20):
    matches = []
    errors = []
    for a in A:
        for b in B:
            they_match, euclidian = points_match(a,b)
            if they_match:
                errors.append(euclidian)          
                matches.append([a[1],a[0]])
                break
    return matches, errors

def corner_detection_2k(img,k1,k2,min_distance=20, debug=False):

    img1 = edge_detection(img,k1)
    img2 = edge_detection(img,k2)

    maxima1 = peak_local_max(img1,min_distance=min_distance)
    maxima2 = peak_local_max(img2,min_distance=min_distance)

    matches, distances = matching_points(maxima1,maxima2,dist=min_distance)

    if debug:
        print("found {} vertical maximas and {} horizontal maximas".format(len(maxima1),len(maxima2)))
        print("found {} matches".format(matches))

    return matches, distances 

def combine_img(img1,img2):
    img1[img2 == True] = True
    return img1
    
def corner_detection(img,min_distance=20):
    maxima = corner_peaks(corner_shi_tomasi(img),min_distance=min_distance)
    maxima = [[x[1],x[0],0] for x in maxima]
    return maxima 

def evaluation(corners,truth_csv,output_json="summary.json",debug=False):
    df = pandas.read_csv(truth_csv)
    df['count'] = 0

    false_pos = 0
    false_neg = 0
    true_pos = 0
    duplicates = 0 
    error = 0.0
    errors = 0
    #for corner in corners:
    #    corner.append(0)

    for _, row in df.iterrows():
        point = [row['x'],row['y']]
        for corner in corners:
            match,distance = points_match(point,corner,20)
            if match:
                corner[2] += 1 
                row['count'] += 1
                error += distance
                errors += 1
        if row['count'] == 0:
            false_neg += 1
        else:
            true_pos += 1
            duplicates += row['count'] -1
            
    for corner in corners:
        if corner[2] == 0:
            false_pos += 1 

    if errors > 0:
        error = error/ errors

    x = {'actual':int(df.shape[0]),
                'detected':int(len(corners)),
                'average distance error':error,
                'false negative':int(false_neg),
                'false positive':int(false_pos),
                'true positive':int(true_pos),
                'duplicate true':int(duplicates)}

    with open(output_json,'w') as fd:
        json.dump(x,fd)
    with open("summary.json",'w') as fd:
        json.dump(x,fd)

    if debug:
        print("Corners Detected:",len(corners))
        print("Corners Actual:",df.shape[0])
        print("False Negatives:",false_neg)
        print("False Positives:",false_pos)
        print("True Positives:",true_pos)
        print("Duplicate True Positives:",duplicates)

if __name__ == "__main__":
    vert_k = np.array([[1,1,1,0,-1,-1,-1],[1,1,1,0,-1,-1,-1],[1,1,1,0,-1,-1,-1],[1,1,1,0,-1,-1,-1],[1,1,1,0,-1,-1,-1],[1,1,1,0,-1,-1,-1],[1,1,1,0,-1,-1,-1]])
    horizontal_k = np.array([[1,1,1,1,1,1,1],[1,1,1,1,1,1,1],[1,1,1,1,1,1,1],[0,0,0,0,0,0,0],[-1,-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1,-1]])
    LoG_k = np.array([\
    [0,1,1,2,2,2,1,1,0],\
    [1,2,4,5,5,5,4,2,1],\
    [1,4,5,3,0,3,5,4,1],\
    [2,5,3,-12,-24,-12,3,5,2],\
    [2,5,0,-24,-40,-24,0,5,2],\
    [2,5,3,-12,-24,-12,3,5,2],\
    [1,4,5,3,0,3,5,4,1],\
    [1,2,4,5,5,5,4,2,1],\
    [0,1,1,2,2,2,1,1,0]\
        ])

    params = yaml.safe_load(open('params.yaml'))['eval']

    orig_img = imread("data/rectangle_2.png",as_gray=True)
    bin_img = binary_img(orig_img)
    

    vert_img = edge_detection(orig_img,vert_k)
    horizontal_img = edge_detection(orig_img,horizontal_k)
    combined_img = combine_img(vert_img,horizontal_img)

    LoG_img = edge_detection(orig_img,LoG_k)

    if params['method'] == 'combined':
        corners = corner_detection(combined_img)
        evaluation(corners,"corners.csv",output_json=params['method']+"_summary.json")
    elif params['method'] == 'LoG':
        corners = corner_detection(LoG_img)
        evaluation(corners,"corners.csv",output_json=params['method']+"_summary.json")
    elif params['method'] == 'premade':
        corners =  corner_peaks(corner_harris(orig_img), min_distance=5, threshold_rel=0.02)
        corners = [[x[1],x[0],0] for x in corners]
        evaluation(corners,"corners.csv",output_json=params['method']+"_summary.json")
