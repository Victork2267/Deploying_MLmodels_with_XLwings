import xlwings as xw
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import pickle
import numpy as np

@xw.func
@xw.arg("X_values", np.array)
def DTC_model(X_values):
    dtc_filename = r"DT_model.pkl"

    with open(dtc_filename, 'rb') as file:
        pickle_model = pickle.load(file)

    y_pred = pickle_model.predict(X_values.reshape(1,-1))
    return y_pred

@xw.func
@xw.arg("X_values", np.array)
def BNC_model(X_values):
    dtc_filename = r"BN_model.pkl"

    with open(dtc_filename, 'rb') as file:
        pickle_model = pickle.load(file)

    y_pred = pickle_model.predict(X_values.reshape(1,-1))
    return y_pred

@xw.func
@xw.arg("X_values", np.array)
def GBC_model(X_values):
    dtc_filename = r"GB_model.pkl"

    with open(dtc_filename, 'rb') as file:
        pickle_model = pickle.load(file)

    y_pred = pickle_model.predict(X_values.reshape(1,-1))
    return y_pred

@xw.func
@xw.arg("X_values", np.array)
def LGC_model(X_values):
    dtc_filename = r"LG_model.pkl"

    with open(dtc_filename, 'rb') as file:
        pickle_model = pickle.load(file)

    y_pred = pickle_model.predict(X_values.reshape(1,-1))
    return y_pred

@xw.func
@xw.arg("X_values", np.array)
def RFC_model(X_values):
    dtc_filename = r"RFC_model.pkl"

    with open(dtc_filename, 'rb') as file:
        pickle_model = pickle.load(file)

    y_pred = pickle_model.predict(X_values.reshape(1,-1))
    return y_pred
