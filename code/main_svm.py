# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 15:50:26 2023

@author: Florent Boxus
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 18:10:59 2023

@author: Florent Boxus
"""
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import init_svm
import metrics
from numba import jit
import sys
from sklearn.metrics import make_scorer
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
import ast
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
def extract_last_coordinate(polyline):
    polyline_list = ast.literal_eval(polyline)  # Convertit la chaîne de texte en une liste Python
    if len(polyline_list) > 0:
        return polyline_list[-1]  # Retourne le dernier tuple de coordonnées
    else:
        return None  # Retourne None si la liste est vide
def extract_first_half_trajectory(polyline):
    polyline_list = ast.literal_eval(polyline)  # Convertit la chaîne de texte en une liste Python
    if len(polyline_list) > 1:
        midpoint = len(polyline_list) // 2
        return str(polyline_list[:midpoint])  # Retourne la première moitié de la trajectoire
    else:
        return None
    
if __name__ == "__main__":
    ############################initializing data
    # Train set
    print('initializing data...') 
    data, n_trip_train = init_svm.load_data("train.csv")
    data = data[data['POLYLINE'] != '[]']#This line filters out rows where the 'POLYLINE' column is an empty list.
    data = data[data['MISSING_DATA'] == False][['TAXI_ID','TIMESTAMP','CALL_TYPE', 'POLYLINE']][:15000]#atkes firsts 100000 rows
    train_data, test_set = train_test_split(data, test_size=0.1, random_state=42)
    print(f"Train data shape: {train_data.shape}")
    # Train set
    # Sélectionnez les 500 premières lignes
    test_set = test_set.head(500)
    
    # Si vous souhaitez également réinitialiser les index après la sélection
    test_set.reset_index(drop=True, inplace=True)
    test_data = test_set.copy()
    
    test_data['POLYLINE'] = test_data['POLYLINE'].apply(extract_first_half_trajectory)
    # Supprimez les lignes où 'POLYLINE' est None (trajectoire avec un seul tuple)
    test_data = test_data.dropna(subset=['POLYLINE'])
    test_data['REAL'] = test_set['POLYLINE'].apply(extract_last_coordinate)
    # Réinitialisez les index après la suppression de lignes
    test_data.reset_index(drop=True, inplace=True)
    print(f"Test data shape: {test_data.shape}")
    test_trips_ids = list(test_data.index)
    n=5
    #########################################################preprocessing training data
    max_length = test_data['POLYLINE'].apply(lambda x: len(x)).max()
    print('max length:',max_length)
    print('preprocessing training data...')
    preprocessed_trained_data=init_svm.preprocessing_training(train_data,n)
    
    ########################################################cluster
    print('Clustering training data')
    train_data,clusterer=init_svm.clustering(preprocessed_trained_data,n)
    cluster_centers = clusterer.cluster_centers_
    print("Clusters coordinates:",cluster_centers)
    
    
    
    train_data['CLUSTER_IN'] = train_data['CLUSTER_IN'].apply(lambda x: np.atleast_1d(np.array(x)))
    
    
    X_train= np.stack(train_data['CLUSTER_IN'].to_numpy())
    y_train=train_data['CLUSTER_OUT']
    
    ####################################################### preeprocessing and clustering test data
    print('preprocessing and clustering test data')
    test_data=init_svm.preprocessing_test(test_data,clusterer,n)
    
    test_data['CLUSTER_IN'] = test_data['CLUSTER_IN'].apply(lambda x: np.atleast_1d(np.array(x)))
    
   
    
    X_test = np.stack(test_data['CLUSTER_IN'].to_numpy())
    ###################################################### solving the model
    #Instantiate and fit the RandomForestClassifier
    print("Fitting the model...")

    """
    SVC=SVC(kernel='linear')
    rf_classifier=RandomForestClassifier(n_estimators=200,random_state=42)
    clf = GradientBoostingClassifier()
    """
 
    print("Training the model...")
    


    clf=SVC(kernel='sigmoid')
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    
    

    y_real=test_data['REAL']
    y_real_array = y_real.to_numpy()

    # Initialise les tableaux pour les latitudes et longitudes
    
    y_latitude = np.empty(len(y_real_array), dtype=float)
    y_longitude = np.empty(len(y_real_array), dtype=float)
    
    # Extrait les latitudes et longitudes des paires de coordonnées
    for i, coord_pair in enumerate(y_real_array):
        y_latitude[i] = coord_pair[1]  # Deuxième coordonnée
        y_longitude[i] = coord_pair[0]  # Première coordonnée
    
    cluster_centers_dict = {label: center for label, center in enumerate(cluster_centers)}
    destinations = np.array([cluster_centers_dict[label] for label in y_pred])
    destinations = np.stack((destinations[:, 1], destinations[:, 0]), axis=-1)
    y_test_stack= np.stack((y_latitude, y_longitude), axis=-1)
    score=metrics.haversine(destinations,y_test_stack)
    print("Mean haversine distance :", np.mean(score))
    init_svm.write_submission(
        trip_ids=test_trips_ids, 
        destinations=destinations,
        file_name="example_submission"
    )
