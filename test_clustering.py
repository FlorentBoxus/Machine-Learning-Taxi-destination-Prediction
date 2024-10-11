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
import init_last_coord
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
    data, n_trip_train = init_last_coord.load_data("train.csv")
    data = data[data['POLYLINE'] != '[]']#This line filters out rows where the 'POLYLINE' column is an empty list.
    data = data[data['MISSING_DATA'] == False][['TAXI_ID','TIMESTAMP','CALL_TYPE', 'POLYLINE']][:100000]#atkes firsts 100000 rows
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
    n=100
    #########################################################preprocessing training data
    max_length = test_data['POLYLINE'].apply(lambda x: len(x)).max()
    print('max length:',max_length)
    print('preprocessing training data...')
    preprocessed_trained_data=init_last_coord.preprocessing_training(train_data,n)
    
    ########################################################cluster
    print('Clustering training data')
    train_data,clusterer=init_last_coord.clustering(preprocessed_trained_data,n)
    cluster_centers = clusterer.cluster_centers_
    print("Clusters coordinates:",cluster_centers)
    
    
    train_data['DAY_WEEK'] = train_data['DAY_WEEK'].apply(lambda x: np.atleast_1d(np.array(x)))
    train_data['WEEK'] = train_data['WEEK'].apply(lambda x: np.atleast_1d(np.array(x)))
    train_data['TIME_DAY'] = train_data['TIME_DAY'].apply(lambda x: np.atleast_1d(np.array(x)))
    train_data['CLUSTER_IN'] = train_data['CLUSTER_IN'].apply(lambda x: np.atleast_1d(np.array(x)))
    train_data['DIRECTION'] = train_data['DIRECTION'].apply(lambda x: np.atleast_1d(np.array(x)))
    training_data = train_data[['WEEK','DAY_WEEK','TIME_DAY','DIRECTION','CLUSTER_IN','CLUSTER_OUT']].copy()
    training_data['FEATURES'] = training_data.apply(lambda row: np.concatenate([row['WEEK'],row['DAY_WEEK'],row['TIME_DAY'], row['DIRECTION'], row['CLUSTER_IN']]), axis=1)
    X_train= np.stack(training_data['FEATURES'].to_numpy())
    y_train=training_data['CLUSTER_OUT']
    
    ####################################################### preeprocessing and clustering test data
    print('preprocessing and clustering test data')
    test_data=init_last_coord.preprocessing_test(test_data,clusterer,n)
    test_data['DAY_WEEK'] = test_data['DAY_WEEK'].apply(lambda x: np.atleast_1d(np.array(x)))
    test_data['WEEK'] = test_data['WEEK'].apply(lambda x: np.atleast_1d(np.array(x)))
    test_data['TIME_DAY'] = test_data['TIME_DAY'].apply(lambda x: np.atleast_1d(np.array(x)))
    test_data['CLUSTER_IN'] = test_data['CLUSTER_IN'].apply(lambda x: np.atleast_1d(np.array(x)))
    test_data['DIRECTION'] = test_data['DIRECTION'].apply(lambda x: np.atleast_1d(np.array(x)))
    testing_data = test_data[['WEEK','DAY_WEEK', 'TIME_DAY','DIRECTION', 'CLUSTER_IN']].copy()
    testing_data['FEATURES'] = testing_data.apply(lambda row: np.concatenate([row['WEEK'],row['DAY_WEEK'],row['TIME_DAY'],row['DIRECTION'], row['CLUSTER_IN']]), axis=1)
    
    X_test = np.stack(testing_data['FEATURES'].to_numpy())
    ###################################################### solving the model
    #Instantiate and fit the RandomForestClassifier
    print("Fitting the model...")

    """
    SVC=SVC(kernel='linear')
    rf_classifier=RandomForestClassifier(n_estimators=200,random_state=42)
    clf = GradientBoostingClassifier()
    """
    """
    print("Training the model...")
    
    param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7,9,None],
}

    # Initialize the GradientBoostingClassifier
    gb_classifier = GradientBoostingClassifier(random_state=42)
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(gb_classifier, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    
    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    
    # Print the best hyperparameters and corresponding accuracy
    best_params = grid_search.best_params_
    best_accuracy = grid_search.best_score_
    print(f'Best Hyperparameters: {best_params}')
    sys.exit()
"""
    
    def grid_score(pred,gt):
        cluster_centers_dict = {label: center for label, center in enumerate(cluster_centers)}
        pred = np.array([cluster_centers_dict[label] for label in pred])
        gt = np.array([cluster_centers_dict[label] for label in gt])
        pred = np.stack((pred[:, 1], pred[:, 0]), axis=-1)
        gt = np.stack((gt[:, 1], gt[:, 0]), axis=-1)
        return -metrics.score(pred,gt)

    """
    
    param_range = {
    'n_estimators': [50, 100,150, 200, 300,400],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30,40,50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2', None]
}"""
    """
    param_range = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.8, 0.9, 1.0],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2', None]
}
    """
    clf=DecisionTreeClassifier()
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
    init_last_coord.write_submission(
        trip_ids=test_trips_ids, 
        destinations=destinations,
        file_name="example_submission"
    )
