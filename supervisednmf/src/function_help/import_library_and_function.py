import pandas as pd
import sys
import json
import numpy as np
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
import re, os
import seaborn as sns
import joblib
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
import pickle
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, classification_report, roc_auc_score, make_scorer, auc, roc_curve
from scipy.stats import pearsonr
from sklearn.svm import SVC, NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shutil
import random
import torch
from xgboost import XGBClassifier
from sklearn.decomposition import NMF
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from collections import Counter
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import *
from xgboost import XGBClassifier
import math
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier
from pathlib import Path
from scipy import stats
import time
import multiprocessing
from sklearn.decomposition import PCA, FastICA
from supervenn import supervenn
from sklearn.feature_selection import SequentialFeatureSelector

from collections import Counter
from numpy import log,dot,exp,shape
from scipy.stats import kendalltau
# import dcor
# import statsmodels.api as sm
from scipy.spatial import ConvexHull
# from shapely.geometry import Polygon, Point
# from umap import UMAP


def read_meta(meta_path):
    meta = pd.read_csv(meta_path)
    return meta

def read_feature(feature_path, meta = None):
    feature = pd.read_csv(feature_path)
    
    if str(meta) != 'None':
        feature = pd.merge(meta[['SampleID']], feature)
    
    return feature
    
def encode_y(y):
    y = np.array([0 if label == 'Control' else 1 for label in y])
    return y

def split_data(feature, meta, cv=5):
    meta = meta[['SampleID', 'Label']]
    
    # Merge the DataFrames on 'SampleID'
    df = pd.merge(meta, feature, on='SampleID')

    # Extract features (all columns except 'SampleID' and 'Label')
    X = df.drop(columns=['SampleID', 'Label'])

    # Extract labels (the 'Label' column)
    y = df['Label']

    X_train, X_test, y_train, y_test = [], [], [], []
    
    if cv <= 1:

        # Split the data into training and testing sets
        X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # save
        X_train.append(X_train_fold)
        X_test.append(X_val_fold)
        y_train.append(y_train_fold)
        y_test.append(y_val_fold)
        
    else:
        
        # StratifiedKFold
        sss = StratifiedKFold(n_splits=cv, shuffle=True, random_state = 42)
        for train_index, val_index in sss.split(meta[['SampleID']], meta[['Label']]):
            # Split the data into training and testing sets
            X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
            y_train_fold, y_val_fold = y[train_index], y[val_index]
            
            # save
            X_train.append(X_train_fold)
            X_test.append(X_val_fold)
            y_train.append(y_train_fold)
            y_test.append(y_val_fold)

    return X_train, X_test, y_train, y_test

def get_cutoff(y_true, y_prob, spec_cutoff=95):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, drop_intermediate = False)
    threshold = thresholds[(fpr <= (100-spec_cutoff)/100).nonzero()[0][-1]]
    threshold = thresholds[(fpr <= (100-spec_cutoff)/100).nonzero()[0][-1]]
    return threshold


# def UMAP_CONVEX_HULL(X_train, y_train, feature_name, output_path, cutoff_outlier = 90):
    
#     palette = {
#       "Control": "green",
#       "Breast": "red",
#       "CRC": "deepskyblue", 
#       "Gastric": "violet",
#       "Liver": "blue",
#       "Lung": "gold",
#       "Target": "gray",
#       "Cancer": "brown",
#       "HCC": "pink",
#       "Ovarian": "yellow",
#       "Duodenal": "black",
#       "Cancer": "red"
#      }
    
#     palette_detail = {
#           "Control": {"color": "green", "linestyle": "-", "lw": 1},
#           "Breast": {"color": "red", "linestyle": "-", "lw": 1},
#           "CRC": {"color": "deepskyblue", "linestyle": "-", "lw": 1}, 
#           "Gastric": {"color": "violet", "linestyle": "-", "lw": 1},
#           "Liver": {"color": "blue", "linestyle": "-", "lw": 1},
#           "Lung": {"color": "gold", "linestyle": "-", "lw": 1},
#           "Target": {"color": "gray", "linestyle": "-", "lw": 1},
#           "Cancer": {"color": "brown", "linestyle": "-", "lw": 1},
#           "HCC": {"color": "pink", "linestyle": "-", "lw": 1},
#           "Ovarian": {"color": "yellow", "linestyle": "-", "lw": 1},
#           "Duodenal": {"color": "black", "linestyle": "-", "lw": 1},
#           "Cancer": {"color": "red", "linestyle": "-", "lw": 1}
#     }
    
#     summary_overlap = []
#     ###################### LOOP CV ###################
#     cv = len(X_train)
#     for fold in range(cv):
#         # Get data at fold
#         X_train_fold, y_train_fold = X_train[fold], y_train[fold]
        
#         ############################################## CREATE 2D UMAP #####################################################
#         df_plot = pd.DataFrame(X_train_fold.copy())
#         df_plot['Label'] = y_train_fold
#         df_plot['Group'] = [label if label == 'Control' else 'Cancer' for label in df_plot['Label']]
#         df_plot = df_plot[['Label', 'Group'] + X_train_fold.columns.tolist()]

#         # Create a sample DataFrame with random data
#         np.random.seed(0)

#         # Perform UMAP projection
#         features = df_plot.iloc[:, 2:]  # Select the feature columns (excluding 'Type' and 'Label')
#         umap_2d = UMAP(n_components=2, init='random', random_state=0)
#         proj_2d = umap_2d.fit_transform(features)

#         # Create a DataFrame from the UMAP projection
#         df_umap = pd.DataFrame(proj_2d, columns=['umap 1', 'umap 2'])
#         df_umap['Group'] = df_plot['Group'].tolist()
#         df_umap['Label'] = df_plot['Label'].tolist()

#         ################################################ UMAP + CONVEX FULL #############################################
#         # Function to plot convex hulls
#         def plot_convex_hull(points, ax, **kwargs):
#             if len(points) > 2:  # ConvexHull requires at least 3 points
#                 hull = ConvexHull(points)
#                 for simplex in hull.simplices:
#                     plt.plot(points[simplex, 0], points[simplex, 1], **kwargs)
#                 return hull
#             return None

#         # Function to calculate points inside or on the edges of a convex hull
#         def points_in_hull_or_on_edges(points, poly):
#             return np.array([poly.intersects(Point(p)) for p in points])

#         # Function to calculate the intersection of two convex hulls
#         def intersect_hulls(hull1, hull2):
#             if hull1 is None or hull2 is None:
#                 return None
#             poly1 = Polygon(hull1.points[hull1.vertices])
#             poly2 = Polygon(hull2.points[hull2.vertices])
#             return poly1.intersection(poly2)

#         hulls = {}
#         hull_show = []
#         overlap_counts = {}

#         cutoff_outlier = 90

#         # Plot UMAP projections with convex hulls
#         plt.figure(figsize=(7, 7))
#         sns.scatterplot(
#             x="umap 1", y="umap 2",
#             hue="Label", s=25,
#             data=df_umap, palette = palette
#         )

#         # Draw convex hulls
#         for t in df_plot['Group'].unique():
#             subset = df_umap[df_umap['Group'] == t]
#             points = subset[['umap 1', 'umap 2']].values

#             # Compute the centroid
#             centroid = points.mean(axis=0)

#             # Calculate distances from the centroid
#             distances = np.linalg.norm(points - centroid, axis=1)

#             # Get the cutoff percentile distance
#             threshold = np.percentile(distances, cutoff_outlier)

#             # Filter out the points beyond the cutoff percentile distance
#             filtered_points = points[distances <= threshold]

#             hull_show.append(t)
#             hull = plot_convex_hull(filtered_points, plt.gca(), **palette_detail[t])
#             hulls[t] = hull

#         plt.xlabel('UMAP 1')
#         plt.ylabel('UMAP 2')
#         plt.legend()


#         ################################################ CACULCATE OVERLAP #############################################

#         set1 = 'Control'
#         set2 = 'Cancer'

#         # Hull1
#         hull1 = hulls[set1]

#         # Hull2
#         hull2 = hulls[set2]

#         # Subset1
#         subset1 = df_umap[df_umap['Group'] == set1][['umap 1', 'umap 2']].values

#         # Remove outliers from subset1
#         centroid1 = subset1.mean(axis=0)
#         distances1 = np.linalg.norm(subset1 - centroid1, axis=1)
#         threshold1 = np.percentile(distances1, cutoff_outlier)
#         subset1_filter = subset1[distances1 <= threshold1]

#         # Subset2
#         subset2 = df_umap[df_umap['Group'] == set2][['umap 1', 'umap 2']].values

#         # Remove outliers from subset2
#         centroid2 = subset2.mean(axis=0)
#         distances2 = np.linalg.norm(subset2 - centroid2, axis=1)
#         threshold2 = np.percentile(distances2, cutoff_outlier)
#         subset2_filter = subset2[distances2 <= threshold2]

#         # print(subset1_filter.shape[0], subset2_filter.shape[0])

#         # Overlap area between two hulls
#         intersection_poly = intersect_hulls(hull1, hull2)

#         points_in_hull1 = points_in_hull_or_on_edges(subset1, intersection_poly).sum()
#         points_in_hull2 = points_in_hull_or_on_edges(subset2, intersection_poly).sum()

#         overlap_counts[(set1, set2)] = (points_in_hull1 + points_in_hull2) / (len(subset1_filter) + len(subset2_filter)) * 100

#         # Print overlap percentages
#         for (set1, set2), ratio in overlap_counts.items():
#             plt.title(f"{feature_name}: UMAP - Convex Hulls: {set1} vs {set2}: {ratio:.0f}%")
#             summary_overlap.append([fold, ratio])

#         plt.savefig('{}/UMAP_fold{}.png'.format(output_path, fold))
#         plt.close()
        
#     summary_overlap = pd.DataFrame(summary_overlap)
#     summary_overlap.columns = ['Fold', 'Similarity score']
#     plt.figure(figsize=(5, 5))
#     sns.lineplot(data = summary_overlap, x = 'Fold', y = 'Similarity score', marker= 'o')
#     plt.ylim([-2, 102])
#     plt.xlim([-0.2, cv-1+0.2])
#     plt.title('{}: Similarity between Control and Cancer (mean = {:.0f}%)'.format(feature_name, summary_overlap['Similarity score'].mean()))
#     plt.ylabel('Score (%)')
#     plt.savefig('{}/Similarity_score.png'.format(output_path))
#     plt.show()

def correlation_with_target(X_train, y_train, output_path):
    correlations = {}
    for i in range(len(X_train)):
        X = X_train[i].copy()
        Y = y_train[i].copy()
        Y = encode_y(Y)
        X['Label'] = Y
        corr = X.corr()['Label'].drop('Label')  # Get correlation with the label, drop the label itself
        correlations[i] = corr

    # Combine the correlations into a single dataframe
    correlation_df = pd.DataFrame(correlations)
    correlation_df['mean'] = np.abs(correlation_df.mean(axis = 1))
    correlation_df = 100 * correlation_df['mean']

    plt.figure(figsize=(5, 5))
    sns.swarmplot(data=correlation_df, s = 5)
    plt.title('Feature Importance Scores')
    plt.ylim([-2, 102])
    plt.ylabel('Correlation with target (%)')
    plt.savefig('{}/Correlation_with_target.png'.format(output_path))
    plt.show()
    
    
def init_NMF_with_NICA(X, rank, lr=0.001, max_iter=50000, tol=1e-8, rowvar=True):
    """Compute the non-negative independent components of the linear generative model x = A * s.

    Here, x is a p-dimensional observable random vector and s is the latent random vector
    of length rank, whose components are statistically independent and non-negative.
    The matrix X is assumed to hold n samples of x, stacked in rows (shape(X) = (n, p)) or
    columns (shape(X) = (p, n)), which can be specified by the rowvar parameter. In practice,
    if shape(X) = (p, n) (resp. shape(X) = (n, p)) this function solves X = A * S
    (resp. X = S.T * A.T) both for S and A, where A is the so-called mixing matrix, with shape
    (p, rank), and S is a (rank, n) matrix which contains n samples of the latent
    source vector, stacked in columns.

    This function implements the method presented in:
    `Blind Separation of Positive Sources by Globally Convergent Gradient Search´
    (https://core.ac.uk/download/pdf/76988305.pdf)

    Args:
        X: Data matrix.
        rank: Dimension of s. Number of latent random variables.
        lr: Learning rate of gradient descent.
        max_iter: Maximum number of iterations of gradient descent.
        tol: Tolerance on update at each iteration.
        rowvar: Whether each row of X corresponds to one of the p variables or not.

    Returns:
        (S, A) if rowvar == True.
        (S.T, A) if rowvar == False.

    """
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################
    def whiten(X, num_components=None, center=True, rowvar=True):
        """Whiten the data in matrix X using PCA decomposition.

        The data corresponds to n samples of a p-dimensional random vector. The shape
        of the matrix can be either (n, p) if each row is considered to be a sample or
        (p, n) if each column is considered to be a sample. How to read the matrix entries
        is specified by the rowvar parameter. Before whitening, a dimensionality reduction
        step can be applied to the data to reduce the p dimensions of each sample to
        num_components dimensions. If num_components is None, the number of dimensions kept
        is the maximum possible (nº of non-zero eigenvalues). For example, if X is full rank
        (rank(X) = min(n, p)), then num_components = p if p < n, and num_components = n-1
        if p >= n.

        Args:
            X: Data matrix.
            num_components: Number of PCA dimensions of the whitened samples.
            center: Whether to center the samples or not (zero-mean whitened samples).
            rowvar: Whether each row of X corresponds to one of the p variables or not.

        Returns:
            (Z, V): The whitened data matrix and the whitening matrix.

        """
        r = num_components

        if rowvar:
            X = X.transpose()

        # Data matrix contains n observations of a p-dimensional vector
        # Each observation is a row of X
        n, p = X.shape

        # Arbitrary (but sensible) choice. In any case, we remove the eigenvectors of 0 eigenvalue later
        if r is None:
            r = min(n, p)

        # Compute the mean of the observations (p-dimensional vector)
        mu = np.mean(X, axis=0)

        # If p > n compute the eigenvectors efficiently
        if p > n:
            # n x n matrix
            M = np.matmul((X-mu), (X-mu).transpose())

            # Eigenvector decomposition
            vals, vecs = np.linalg.eig(M)
            vals, vecs = vals.real, vecs.real

            # Sort the eigenvectors by "importance" and get the first r
            pairs = sorted([(vals[i], vecs[:, i]) for i in range(len(vals))], key=lambda x: x[0], reverse=True)
            pairs = [p for p in pairs if abs(p[0]) > 1e-10]  # Remove the eigenvectors of 0 eigenvalue
            pairs = pairs[:r]

            # nxr matrix of eigenvectors (each column is an n-dimensional eigenvector)
            E = np.array([p[1] for p in pairs]).transpose()

            # pxr matrix of the first r eigenvectors of the covariance of X
            # Note that we normalize!
            E = np.matmul((X-mu).transpose(), E)
            E /= np.linalg.norm(E, axis=0)

            # Eigenvalues of cov(X) to the -1/2
            # Note that we rescale the eigenvalues of M to get the eigenvalues of cov(X)!
            diag = np.array([1/np.sqrt(p[0]/(n-1)) for p in pairs])

        else:
            # p x p matrix
            C = np.cov(X, rowvar=False)

            # Eigenvector decomposition
            vals, vecs = np.linalg.eig(C)
            vals, vecs = vals.real, vecs.real

            # Sort the eigenvectors by "importance" and get the first r
            pairs = sorted([(vals[i], vecs[:, i]) for i in range(len(vals))], key=lambda x: x[0], reverse=True)
            pairs = [p for p in pairs if abs(p[0]) > 1e-10]  # Remove the eigenvectors of 0 eigenvalue
            pairs = pairs[:r]

            # pxr matrix of the first r eigenvectors of the covariance of X
            E = np.array([p[1] for p in pairs]).transpose()

            # Eigenvalues of cov(X) to the -1/2
            diag = np.array([1/np.sqrt(p[0]) for p in pairs])

        # Warn that the specified number of components is larger
        # than the number of non-zero eigenvalues.
        if num_components is not None:
            if num_components > len(pairs):
                warnings.warn(
                    'The desired number of components (%d) is larger than the actual dimension'
                    ' of the PCA subespace (%d)' % (num_components, len(pairs))
                )

        # Center and whiten the data
        if center:
            X = X - mu

        # Whitening matrix
        V = E * diag

        # White data
        Z = np.matmul(X, V)

        if rowvar:
            Z = Z.transpose()

        # Since X is assumed to be (n, p) through the computations, the current
        # whitening matrix V is in fact the transpose of the actual whitening matrix.
        # Observation: If z = V * x for random column vectors x, z, then Z = X * V
        # for the (n, p) and (n, r) matrices X, Z of observations of x, z.
        V = V.transpose()

        return Z, V
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################
    
    # Whiten the data
    Z, V = whiten(X, rank, center=False, rowvar=rowvar)

    if rank > V.shape[0]:
        warnings.warn(
            'The desired number of sources (%d) is larger than the actual dimension'
            ' of the whitened observable random vector (%d). The number of sources'
            ' will be set to %d' % (rank, V.shape[0], V.shape[0])
        )
        rank = V.shape[0]

    # We assume rowvar is True throughout the algorithm
    if not rowvar:
        Z = Z.transpose()

    # Initialize W
    W = np.eye(rank)

    for i in range(max_iter):
        W0 = W

        # Compute gradient
        Y = np.matmul(W, Z)
        f = np.minimum(0, Y)
        f_Y = np.matmul(f, Y.transpose())
        E = (f_Y - f_Y.transpose()) / Y.shape[1]

        # Gradient descent
        W -= lr * np.matmul(E, W)

        # Symmetric orthogonalization
        M = np.matmul(W, W.transpose())
        vals, vecs = np.linalg.eig(M)
        vals, vecs = vals.real, vecs.real

        W_sqrt = vecs / np.sqrt(vals)
        W_sqrt = np.matmul(W_sqrt, vecs.transpose())
        W = np.matmul(W_sqrt, W)

        print(i, np.linalg.norm(W - W0))
        if np.linalg.norm(W - W0) < tol:
            break

    # Independent sources (up to an unknown permutation y = Q * s)
    Y = np.matmul(W, Z)

    # Compute the mixing matrix A' = A * Q.T
    # (which is A up to a permutation of its columns)
    # from the identity y = Q * s = W * V * A * s.
    # It then holds x = A * s = A * Q.T * y = A' * y.
    # Note: A' is computed as the right Moore-Penrose
    # inverse of W * V, but A' may not be unique since
    # in general p != rank and any right inverse
    # could be taken as A'.
    WV = np.matmul(W, V)
    WV_ = np.matmul(WV, WV.transpose())
    WV_ = np.linalg.inv(WV_)
    WV_ = np.matmul(WV.transpose(), WV_)

    if not rowvar:
        Y = Y.transpose()

    Y = np.abs(Y)
    WV_ = np.abs(WV_)
    
    return WV_, Y


def init_NMF_with_FICA(X, rank):
    from sklearn.decomposition import FastICA
    ica = FastICA(n_components=rank)
    S = ica.fit_transform(X)  # Reconstruct signals
    A = ica.mixing_.T  # Get estimated mixing matrix
    
    S = np.abs(S)
    A = np.abs(A)
    return S, A

def init_NMF(X, rank, init_mode):
    class CustomNMF(NMF):

        ############################ START MODIFY ##############################
        def init_nmf(self, X):
            X = self._validate_data(
                X, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32], reset=False
            )

            self._check_params(X)
            
            # initialize W and H
            W = None
            H = None
            update_H = True
            W, H = self._check_w_h(X, W, H, update_H)

            return W, H
        ############################ END MODIFY #############################

    nmf = CustomNMF(n_components=rank, init=init_mode, random_state=42)
    W, H = nmf.init_nmf(X)
    
    return W, H

def init_W(X, rank):
    n_samples, n_features = X.shape
    avg = np.sqrt(X.mean() / rank)
    W = np.full((n_samples, rank), avg, dtype=X.dtype)
    return W