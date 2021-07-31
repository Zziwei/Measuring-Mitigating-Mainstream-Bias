import tensorflow as tf
import time
import numpy as np
import pickle
import argparse
import pandas as pd
import utility
from tqdm import tqdm
from scipy.sparse import coo_matrix


np.random.seed(1)
tf.set_random_seed(1)


def dense_layer(x, dim, scope):
    with tf.variable_scope(scope):
        init = tf.truncated_normal_initializer(stddev=0.1)
        W = tf.get_variable(scope + '_w', shape=[x.get_shape().as_list()[1], dim], initializer=init)
        ans = tf.matmul(x, W)
        return ans, tf.reduce_sum(W ** 2) ** 0.5


def distribution_plot(x, y, bins=20):
    x_max = np.max(x)
    x_min = np.min(x)
    step = (x_max - x_min) / bins
    x_array = []
    mean_array = []
    std_array = []
    for i in range(bins):
        start = x_min + step * i
        end = x_min + step * (i + 1)
        x_array.append((start + end) / 2)
        tmp = y[np.where((x >= start) & (x <= end))[0]]
        mean_array.append(np.mean(tmp) if len(tmp) > 0 else 0)
        std_array.append(np.std(tmp) if len(tmp) > 0 else 0)
    print(mean_array)


class DeepSVDD:
    def __init__(self, sess, args, train_df, ndcg):

        self.ndcg = ndcg

        self.expert = args.expert

        self.sess = sess
        self.args = args

        self.num_item = args.num_item
        self.num_user = args.num_user

        self.layers = args.layers
        self.dimension = self.layers[-1]

        self.batch_size = args.bs

        self.epoch = args.epoch

        self.lr = args.lr  # learning rate
        self.reg = args.reg  # regularization term trade-off

        row_array = train_df['userId'].values.reshape(-1)
        col_array = train_df['itemId'].values.reshape(-1)
        self.train_mat = coo_matrix((np.ones(row_array.shape[0]), (row_array, col_array)),
                                    shape=(self.num_user, self.num_item)).toarray()

        print(self.args)
        self._prepare_model()

    def run(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)

        embeddings = self.sess.run(self.embeddings, feed_dict={self.input_mat: self.train_mat})

        self.center_array = np.mean(embeddings, axis=0, keepdims=True)

        for epoch_itr in range(1, self.epoch + 1):
            self.train_model(epoch_itr)
            self.test_model()

    def _prepare_model(self):

        self.input_mat = tf.placeholder(dtype=tf.float32, shape=[None, self.num_item], name="input_mat")
        self.input_center = tf.placeholder(dtype=tf.float32, shape=[1, self.dimension], name="input_center")

        self.reg_loss = 0

        gate, _ = dense_layer(self.input_mat, self.expert, 'gate_layer')
        gate = tf.nn.tanh(gate)

        expert_list = []
        for e in range(self.expert):
            tmp_layer = self.input_mat
            for i in range(len(self.layers)):
                layer_dim = self.layers[i]
                tmp_layer, tmp_reg = dense_layer(tmp_layer, layer_dim, 'expert_' + str(e) + 'W_' + str(i))
                self.reg_loss += tmp_reg
            expert_list.append(tf.reshape(tmp_layer, [-1, 1, self.dimension]))
        expert_concat = tf.concat(expert_list, 1)  # size: batch_size X expert X self.dimension
        expert_concat = tf.linalg.matmul(tf.reshape(gate, [-1, 1, self.expert]), expert_concat)
        self.embeddings = tf.reshape(expert_concat, [-1, self.dimension])  # size: batch_size X self.dimension

        self.embeddings, tmp_reg = dense_layer(self.embeddings, self.dimension, 'final_layer')
        self.reg_loss += tmp_reg
        self.embeddings /= tf.reduce_mean(self.embeddings ** 2, axis=1, keepdims=True) ** 0.5

        self.cost1 = tf.reduce_mean(tf.reduce_sum((self.embeddings - self.input_center) ** 2, axis=1))
        self.cost2 = self.reg * self.reg_loss

        self.cost = self.cost1 + self.cost2

        with tf.variable_scope("Optimizer", reuse=tf.AUTO_REUSE):
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.optimizer = optimizer.minimize(self.cost)

    def train_model(self, itr):
        epoch_cost = 0.
        epoch_cost1 = 0.
        epoch_cost2 = 0.

        num_batch = int(self.num_user / float(self.batch_size)) + 1
        random_idx = np.random.permutation(self.num_user)
        for i in tqdm(range(num_batch)):
            batch_idx = None
            if i == num_batch - 1:
                batch_idx = random_idx[i * self.batch_size:]
            elif i < num_batch - 1:
                batch_idx = random_idx[(i * self.batch_size):((i + 1) * self.batch_size)]
            _, tmp_cost, tmp_cost1, tmp_cost2 = self.sess.run(  # do the optimization by the minibatch
                [self.optimizer, self.cost, self.cost1, self.cost2],
                feed_dict={self.input_mat: self.train_mat[batch_idx, :],
                           self.input_center: self.center_array})
            epoch_cost += tmp_cost
            epoch_cost1 += tmp_cost1
            epoch_cost2 += tmp_cost2

        print("Training //", "Epoch %d //" % itr, " Total cost = {:.5f}".format(epoch_cost),
              " Total cost1 = {:.5f}".format(epoch_cost1), " Total cost2 = {:.5f}".format(epoch_cost2))

    def test_model(self):
        embeddings = self.sess.run(self.embeddings, feed_dict={self.input_mat: self.train_mat})
        distance = np.sum((embeddings - self.center_array) ** 2, axis=1) ** 0.5

        sort_idx = np.argsort(distance)[-1::-1]
        ndcg_sort = self.ndcg[sort_idx]
        distribution_plot(np.arange(self.num_user), ndcg_sort, bins=5)

        with open('./Data/' + self.args.data + '/MS_DeepSVDD.npy', "wb") as f:
            np.save(f, -distance)

    @staticmethod
    def l2_norm(tensor):
        return tf.reduce_sum(tf.square(tensor))


parser = argparse.ArgumentParser(description='DeepSVDD')
parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train')
parser.add_argument('--layers', nargs='+', type=int, default=[400, 300])
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--reg', type=float, default=0.01, help='regularization')
parser.add_argument('--bs', type=int, default=200, help='batch size')
parser.add_argument('--expert', type=int, default=10)
parser.add_argument('--data', type=str, default='ML1M', help='path to eval in the downloaded folder')

args = parser.parse_args()

with open('./Data/' + args.data + '/info.pkl', 'rb') as f:
    info = pickle.load(f)
    args.num_user = info['num_user']
    args.num_item = info['num_item']

train_df = pd.read_csv('./Data/' + args.data + '/train_df.csv')

train_like = list(np.load('./Data/' + args.data + '/user_train_like.npy', allow_pickle=True))
test_like = list(np.load('./Data/' + args.data + '/user_test_like.npy', allow_pickle=True))
Rec = np.load('./Data/' + args.data + '/Rec_VAE.npy')

user_precision = []
user_recall = []
user_ndcg = []
for u in range(args.num_user):
    Rec[u, train_like[u]] = -100000.0

for u in tqdm(range(args.num_user)):
    scores = Rec[u, :]
    top_iid = np.argpartition(scores, -20)[-20:]
    top_iid = top_iid[np.argsort(scores[top_iid])[-1::-1]]

    # calculate the metrics
    if not len(test_like[u]) == 0:
        precision_u, recall_u, ndcg_u = utility.user_precision_recall_ndcg(top_iid, test_like[u])
    else:
        precision_u = recall_u = ndcg_u = [-1, -1, -1, -1]
    user_precision.append(precision_u)
    user_recall.append(recall_u)
    user_ndcg.append(ndcg_u)
ndcg = np.array(user_ndcg)[:, 3]

print('!' * 100)

with tf.Session() as sess:
    autorec = DeepSVDD(sess, args, train_df, ndcg)
    autorec.run()
