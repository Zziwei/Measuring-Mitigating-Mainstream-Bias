import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from math import log
from scipy.sparse import coo_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats import mode
import utility
import argparse
from sklearn.neighbors import LocalOutlierFactor


parser = argparse.ArgumentParser(description='Mainstream_Scores')
parser.add_argument('--data', type=str, default='ML1M', help='path to eval in the downloaded folder')

args = parser.parse_args()

with open('./Data/' + args.data + '/info.pkl', 'rb') as f:
    info = pickle.load(f)
    args.num_user = info['num_user']
    args.num_item = info['num_item']


with open('./Data/' + args.data + '/info.pkl', 'rb') as f:
    info = pickle.load(f)
    num_user = info['num_user']
    num_item = info['num_item']

train_df = pd.read_csv('./Data/' + args.data + '/train_df.csv')

pos_user_array = train_df['userId'].values
pos_item_array = train_df['itemId'].values
train_mat = coo_matrix((np.ones(len(pos_user_array)), (pos_user_array, pos_item_array)), shape=(num_user, num_item)).toarray()
user_pop = np.sum(train_mat, axis=1)

Jaccard_mat = np.matmul(train_mat, train_mat.T)
deno = user_pop.reshape((-1, 1)) + user_pop.reshape((1, -1)) - Jaccard_mat + 1e-7
Jaccard_mat /= deno
Jaccard_mat = Jaccard_mat + np.eye(num_user) * -9999
Jaccard_mat = Jaccard_mat[np.where(Jaccard_mat > -1)].reshape((num_user, num_user - 1))
MS_similarity = np.mean(Jaccard_mat, axis=1)
with open('./Data/' + args.data + '/MS_similarity.npy', "wb") as f:
    np.save(f, MS_similarity)

avg_user = np.mean(train_mat, axis=0)
MS_distribution = np.matmul(train_mat, avg_user.reshape((-1, 1))).reshape(-1)
deno1 = np.sum(train_mat ** 2, axis=1) ** 0.5
deno2 = np.sum(avg_user ** 2) ** 0.5
MS_distribution = MS_distribution / deno1 / deno2
with open('./Data/' + args.data + '/MS_distribution.npy', "wb") as f:
    np.save(f, MS_distribution)

clf = LocalOutlierFactor(n_neighbors=300, n_jobs=10)
clf.fit(train_mat)
MS_density = -clf.negative_outlier_factor_
with open('./Data/' + args.data + '/MS_density.npy', "wb") as f:
    np.save(f, MS_density)
