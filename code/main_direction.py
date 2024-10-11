# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 08:53:23 2023

@author: Florent Boxus
"""

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import init_submit
import metrics
from numba import jit
import init_last_coord
import sys
import ast
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier



def extract_first_half_trajectory(polyline):#extract  first half of the coordinates
    polyline_list = ast.literal_eval(polyline)  # Convertit la chaîne de texte en une liste Python
    if len(polyline_list) > 1:
        midpoint = len(polyline_list) // 2
        return str(polyline_list[:midpoint])  # Retourne la première moitié de la trajectoire
    else:
    
        return str(polyline_list)
    
    
def extract_last_coordinate(polyline):#extract the 
    polyline_list = ast.literal_eval(polyline)  # Convertit la chaîne de texte en une liste Python
    if len(polyline_list) > 0:
        return polyline_list[-1]  # Retourne le dernier tuple de coordonnées
    else:
        return None  # Retourne None si la liste est vide
    
    
    
if __name__ == "__main__":
    ############################initializing data

    m=10#parameter deining the threshold 
    
    print('initializing data...') 
    data, n_trip_train = init_submit.load_data("train.csv")
    data = data[data['POLYLINE'] != '[]']
    data = data[data['MISSING_DATA'] == False][['TAXI_ID','TIMESTAMP','CALL_TYPE', 'POLYLINE']][:10000]#takes first 10000 rows
    train_data, test_set = train_test_split(data, test_size=0.1, random_state=42)#fix the random state so we always have the same test set
    print(f"Train data shape: {train_data.shape}")

    # Sélectionne les 500 premières lignes
    test_set = test_set.head(500)
    
    # Reset des indexs
    test_set.reset_index(drop=True, inplace=True)
    test_set['REAL'] = test_set['POLYLINE'].apply(extract_last_coordinate)#this will be the destinations we will want to predict
    test_set['POLYLINE'] = test_set['POLYLINE'].apply(extract_first_half_trajectory)
    # Supprimez les lignes où 'POLYLINE' est None (trajectoire avec un seul tuple)

    print(f"Test data shape: {test_set.shape}")
    test_trips_ids = list(test_set.index)
    

    ###############################################################################
    print('Preprocessing training set...')
    cleaned_data=init_submit.preprocessing_training(train_data)
    print('clustering training set...')
    cluster_centers=init_submit.clustering(cleaned_data)
    print(cluster_centers)
    print('Preprocessing test set...')
    #test_data2=init_submit.preprocessing_testing(test_set)
    delta=0.86#threshold value
    
    prediction=np.zeros((test_set.shape[0],2))
    print('Algo execution begins...')
    
    for i in range(0,test_set.shape[0],1):
        trajectory=test_set['POLYLINE'].iloc[i]
        trajectory = ast.literal_eval(trajectory)
        coordinates = np.array(trajectory)
        if coordinates.shape[0]<m:
            
            prediction[i]=coordinates[len(coordinates)-2]#in this case we return the  second to last coordinate as a prediction
            continue
        else:
            coordinates_list=[]#epsilon in the documentation
            start_long=coordinates[0][0]*np.pi/180#converts the coordinates in radians
            start_lat=coordinates[0][1]*np.pi/180
            end_long=coordinates[coordinates.shape[0]-1][0]*np.pi/180
            end_lat=coordinates[coordinates.shape[0]-1][1]*np.pi/180
            theta_traj=init_submit.compute_direction(start_long,start_lat,end_long,end_lat)
            for j in range(0,len(cluster_centers),1):
                end_long=cluster_centers[j][0]*np.pi/180
                end_lat=cluster_centers[j][1]*np.pi/180
                theta=init_submit.compute_direction(start_long,start_lat,end_long,end_lat)
                bearing_error=init_submit.bearing_err(theta_traj, theta)
                if bearing_error<delta:
                    coordinates_list.append((cluster_centers[j][0],cluster_centers[j][1]))
            coordinates_array = np.array(coordinates_list)
            prediction[i]=np.mean(coordinates_array, axis=0)
            
    ###############################################################################################

    y_real=test_set['REAL']
    destinations = np.stack((prediction[:, 1], prediction[:, 0]), axis=-1)#shapes the results to be scored
    y_real_array = y_real.to_numpy()
    print(destinations)

    y_latitude = np.empty(len(y_real_array), dtype=float)
    y_longitude = np.empty(len(y_real_array), dtype=float)
    

    for i, coord_pair in enumerate(y_real_array):
        y_latitude[i] = coord_pair[1]  # Deuxième coordonnée
        y_longitude[i] = coord_pair[0]  # Première coordonnée
    y_test_stack= np.stack((y_latitude, y_longitude), axis=-1)
    score=metrics.haversine(destinations,y_test_stack)
    print("Mean haversine distance 1:", np.mean(score))
