# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 08:53:10 2023

@author: Florent Boxus
"""
import numpy as np
import pandas as pd
import sys
from sklearn.tree import DecisionTreeRegressor
import metrics
import ast
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math

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


def smoothing(trajectory):
    """
    

    Parameters
    ----------
    trajectory : List of strings
        Corresponding to tuples of coordinates

    Returns
    -------
    smoothed_trajectory : np.array
        Trajectory where outliers were removed

    """
    trajectory = ast.literal_eval(trajectory)
    coordinates = np.array(trajectory)
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

def preprocessing_training(data):
    """
    

    Parameters
    ----------
    data : panda datadrame
        

    Returns
    -------
    data : panda dataframe
        The outliers have been removed and the destination extracted

    """
    data['POLYLINE'] = data['POLYLINE'].apply(smoothing)
    data['END_DEST'] = data['POLYLINE'].apply(lambda x: x[-1])
    
    
    return data


def compute_direction(start_long,start_lat,end_long,end_lat):
    """
    

    Parameters
    ----------
    start_long : float
        start longitude in radians
    start_lat : float
        start latitude in radians
    end_long : float
        end longitude in  radians
    end_lat : float
        end latitude in radians

    Returns
    -------
    theta : float
        Direction of the trip in radians

    """

    y=np.sin(end_long-start_long)*np.cos(end_lat)
    x=np.sin(end_lat)-np.sin(start_lat)*np.cos(start_lat)*np.cos(end_long-start_long)
    theta=compute_atan2(y,x)
    return theta

def compute_atan2(y,x):
    """
    

    Parameters
    ----------
    y : float
    x : float


    Returns
    -------
    TYPE: float 

    """
    sigma=np.tan(y/x)
    if x>0:
        return sigma* math.copysign(1, y)
    elif x==0:
        return (np.pi/2)*math.copysign(1, y)
    else:
        return (np.pi-sigma)*math.copysign(1, y)
    
    
def bearing_err(theta1,theta2):#computes the bearing error
    
    return abs(np.sin((theta2-theta1)/2))#assume directions are in radian


def clustering(data):
    k_clusters = 1500
    end_dest_array = data['END_DEST'].to_numpy()
    out_coordinates= np.vstack(end_dest_array)
    kmeans = KMeans(n_init=10,n_clusters=k_clusters, random_state=42)
    kmeans.fit(out_coordinates)
    cluster_centers = kmeans.cluster_centers_
    return cluster_centers
