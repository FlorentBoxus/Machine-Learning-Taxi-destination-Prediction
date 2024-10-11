# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 15:50:17 2023

@author: Florent Boxus
"""

# -*- coding: utf-8 -*-

import math
import random
import numpy as np
import pandas as pd
import sys
from sklearn.tree import DecisionTreeRegressor
import metrics
import ast
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
def write_submission(trip_ids, destinations, file_name="submission"):
    """
    This function writes a submission csv file given the trip ids, 
    and the predicted destinations.

    Parameters
    ----------
    trip_id : List of Strings
        List of trip ids (e.g., "T1").
    destinations : NumPy Array of Shape (n_samples, 2) with float values
        Array of destinations (latitude and longitude) for each trip.
    file_name : String
        Name of the submission file to be saved.
        Default: "submission".
    """
    n_samples = len(trip_ids)
    assert destinations.shape == (n_samples, 2)#. It ensures that destinations is a 2D array with a shape of (n_samples, 2)

    submission = pd.DataFrame(#This specifies the data to be included in the DataFrame. The 'LATITUDE' column is populated with the values from the first column of the destinations array, and the 'LONGITUDE' column is populated with the values from the second column.
        data={
            'LATITUDE': destinations[:, 0],
            'LONGITUDE': destinations[:, 1],
        },
        columns=["LATITUDE", "LONGITUDE"],#This sets the order of columns in the DataFrame.
        index=trip_ids,#This assigns the trip_ids as the index for the DataFrame, associating each row of data with its corresponding trip identifie
    )

    # Write file
    submission.to_csv(file_name + ".csv", index_label="TRIP_ID")


def load_data(csv_path):
    """
    Reads a CSV file (train or test) and returns the data contained.

    Parameters
    ----------
    csv_path : String
        Path to the CSV file to be read.
        e.g., "train.csv"

    Returns
    -------
    data : Pandas DataFrame 
        Data read from CSV file.
    n_samples : Integer
        Number of rows (samples) in the dataset.
    """
    data = pd.read_csv(csv_path, index_col="TRIP_ID")# The index_col="TRIP_ID" parameter indicates that the 'TRIP_ID' column should be used as the index for the resulting DataFrame

    return data, len(data)#In summary, the load_data function reads a CSV file, sets the 'TRIP_ID' column as the index of the DataFrame, and returns both the DataFrame and the number of samples in the dataset.


def dummy_preprocessing(data):
    data = data[data['POLYLINE'] != '[]']#This line filters out rows where the 'POLYLINE' column is an empty list.
    data = data[data['MISSING_DATA'] == False][['CALL_TYPE', 'DAY_TYPE', 'POLYLINE']][:50000]#This line filters out rows where 'MISSING_DATA' is False and selects only the specified columns ('CALL_TYPE', 'DAY_TYPE', 'POLYLINE') from the DataFrame. The [:10000] part limits the DataFrame to the first 10,000 rows
    data['CALL_TYPE'] = data['CALL_TYPE'].replace(['A', 'B', 'C'], [0, 1, 2])#This line replaces the values in the 'CALL_TYPE' column, mapping 'A' to 0, 'B' to 1, and 'C' to 2
    data['DAY_TYPE'] = data['DAY_TYPE'].replace(['A', 'B', 'C'], [0, 1, 2])#Similarly, this line replaces values in the 'DAY_TYPE' column using the same mapping.
    data['START_Long'] = [eval(polyline)[0][0] for polyline in data['POLYLINE']]#The next four lines extract the start and end longitude and latitude from the 'POLYLINE' column using the eval function. It assumes that 'POLYLINE' contains a list of coordinates, and it extracts the first and last coordinates for each row
    data['START_Lat'] = [eval(polyline)[0][1] for polyline in data['POLYLINE']]
    data['END_Long'] = [eval(polyline)[-1][0] for polyline in data['POLYLINE']]
    data['END_Lat'] = [eval(polyline)[-1][1] for polyline in data['POLYLINE']]
    #data = data.drop('POLYLINE', axis=1)#This line drops the original 'POLYLINE' column from the DataFrame.

    X, y = data.drop(['END_Long', 'END_Lat'], axis=1), data[['END_Long', 'END_Lat']]#This line separates the features (X) and the target values (y) from the modified DataFrame.

    return X, y
def smoothing(trajectory):
    """
    Removes outliers
    """
    coordinates=trajectory
    mean_latitude = np.mean(coordinates[:, 1])
    std_latitude = np.std(coordinates[:, 1])

    latitude_good = np.abs(coordinates[:, 1] - mean_latitude) <= 1.5 * std_latitude
    coordinates = coordinates[latitude_good]

    mean_longitude = np.mean(coordinates[:, 0])
    std_longitude = np.std(coordinates[:, 0])

    longitude_mask = np.abs(coordinates[:, 0] - mean_longitude) <= 3 * std_longitude
    coordinates = coordinates[longitude_mask]

    # Combine latitude and longitude back into a NumPy array of coordinate tuples
    smoothed_trajectory = coordinates.tolist()

    return smoothed_trajectory

    
def preprocessing_training(data,max_length):
    """
    Preprocesses the training data by extracting features of interest
    """
    #data['POLYLINE'] = data['POLYLINE'].apply(smoothing)
   
    data['POLYLINE'] = data['POLYLINE'].apply(ast.literal_eval)
    # Convert lists in numpy arrays
    data['POLYLINE'] = data['POLYLINE'].apply(np.array)
    data['POLYLINE'] = data['POLYLINE'].apply(smoothing)
    data['END_DEST'] = data['POLYLINE'].apply(lambda x: x[-1])

    def create_features(trajectory):#fonction impbriquée permet d'utiliser maxdepth sans le passer en argument avec apply
        if len(trajectory)==1:
            padding = np.tile(trajectory[0], (2*max_length  -len(trajectory), 1))
            features = np.concatenate((padding, trajectory))# do not include last element--> used in end_dest
            return features
        elif len(trajectory)<2*max_length:
            padding=np.tile(trajectory[math.floor(len(trajectory)/2)],(2*max_length  -len(trajectory),1))
            return np.concatenate((trajectory[0:math.floor(len(trajectory)/2)], padding, trajectory[len(trajectory)-math.ceil(len(trajectory)/2):len(trajectory)]))
        else:

            return np.concatenate((trajectory[0:max_length],trajectory[len(trajectory)-2-max_length:len(trajectory)-2]))

    data['TRAJ_FEATURES'] = data['POLYLINE'].apply(create_features)
    return data
def clustering(data,max_length):
    """
    Clusters the training data
    """
    data['TRAJ_FEATURES'] = data['TRAJ_FEATURES'].apply(lambda x: list(x))
    all_coordinates = np.concatenate(data['TRAJ_FEATURES'].tolist())
    k_clusters = 15
    end_dest_array = data['END_DEST'].to_numpy()
    out_coordinates= np.vstack(end_dest_array)
    
    k_values = np.arange(5, 30,1)  # Choose a range of K values
    intra_variances = []
    
    for k in k_values:
        kmeans_ = KMeans(n_init=10,n_clusters=k)
        kmeans_.fit(out_coordinates)
        intra_variances.append(kmeans_.inertia_)
    plt.plot(k_values, intra_variances, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Intra-Variance')
    plt.title('Intra-Variance vs. Number of Clusters')
    plt.show()
    
    kmeans = KMeans(n_init=10,n_clusters=k_clusters, random_state=42)
    kmeans.fit(out_coordinates)
    
    labels= kmeans.predict(all_coordinates)
    labels_out=kmeans.predict(out_coordinates)
    labels_array=np.zeros((len(data['TRAJ_FEATURES']),2*max_length))
    print(labels.shape)
    print(labels_array.shape)
    for i in range(0,len(data['TRAJ_FEATURES']),1):
        labels_array[i]=labels[2*max_length*i:2*max_length*i+2*max_length]
    label_list = labels_array.tolist()
    data['CLUSTER_IN'] = label_list
    data['CLUSTER_OUT']=labels_out
    return data,kmeans

def preprocessing_test(data,clusterer,max_length):
   """
   Preprocesses the test set
   """
    data['POLYLINE'] = data['POLYLINE'].apply(ast.literal_eval)
    data['POLYLINE'] = data['POLYLINE'].apply(np.array)
    def create_features(trajectory):#fonction impbriquée permet d'utiliser maxdepth sans le passer en argument avec apply
        if len(trajectory)==1:
            padding = np.tile(trajectory[0], (2*max_length  -len(trajectory), 1))
            features = np.concatenate((padding, trajectory))# do not include last element--> used in end_dest
            return features
        elif len(trajectory)<2*max_length:
            padding=np.tile(trajectory[math.floor(len(trajectory)/2)],(2*max_length  -len(trajectory),1))
            return np.concatenate((trajectory[0:math.floor(len(trajectory)/2)], padding, trajectory[len(trajectory)-math.ceil(len(trajectory)/2):len(trajectory)]))
        else:

            return np.concatenate((trajectory[0:max_length],trajectory[len(trajectory)-max_length:len(trajectory)]))


    data['TRAJ_FEATURES'] = data['POLYLINE'].apply(create_features)
    
    all_coordinates = np.concatenate(data['TRAJ_FEATURES'].tolist())
    labels= clusterer.predict(all_coordinates)
    labels_array=np.zeros((len(data['TRAJ_FEATURES']),2*max_length))
    print(data['TRAJ_FEATURES'].shape)
    print(labels_array.shape)
    for i in range(0,len(data['TRAJ_FEATURES']),1):
        labels_array[i]=labels[2*max_length*i:2*max_length*i+2*max_length]
    label_list = labels_array.tolist()
    data['CLUSTER_IN'] = label_list
    return data