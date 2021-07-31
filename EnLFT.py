from time import time
import numpy as np
import pickle
import argparse
import pandas as pd
import utility
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy
from scipy.sparse import csr_matrix
from multiprocessing import Process, Queue, Pool, Manager
import utility
from scipy.sparse import coo_matrix
from torch.utils.data import DataLoader
import concurrent.futures
from concurrent import futures
# import torch.multiprocessing as multiprocessing

import warnings

warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

np.random.seed(1)
torch.random.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)


class VAE(nn.Module):
    def __init__(self, args, train_df, device):
        super(VAE, self).__init__()

        self.num_item = args.num_item
        self.num_user = args.num_user

        train_df = train_df
        user_array = train_df['userId'].values.reshape(-1)
        item_array = train_df['itemId'].values.reshape(-1)
        self.train_matrix = csr_matrix((np.ones(len(train_df)), (user_array, item_array)), shape=(self.num_user, self.num_item))

        self.enc_dims = [self.num_item, args.hidden]
        self.dec_dims = self.enc_dims[::-1]
        self.dims = self.enc_dims + self.dec_dims[1:]

        self.data = args.data

        self.lr = args.lr  # learning rate
        self.reg = args.reg  # regularization term trade-off
        self.dropout = args.dropout

        self.anneal = args.anneal

        self.device = device

        self.build_graph()

    def predict_all(self):
        R = self.predict(np.arange(self.num_user))
        return R

    def build_graph(self):
        self.encoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.enc_dims[:-1], self.enc_dims[1:])):
            if i == len(self.enc_dims[:-1]) - 1:
                d_out *= 2
            self.encoder.append(nn.Linear(d_in, d_out))
            if i != len(self.enc_dims[:-1]) - 1:
                self.encoder.append(nn.Tanh())

        self.decoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.dec_dims[:-1], self.dec_dims[1:])):
            self.decoder.append(nn.Linear(d_in, d_out))
            if i != len(self.dec_dims[:-1]) - 1:
                self.decoder.append(nn.Tanh())

        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)

        # Send model to device (cpu or gpu)
        self.to(self.device)

    def forward(self, x):
        # encoder
        h = F.dropout(F.normalize(x), p=self.dropout, training=self.training)
        for layer in self.encoder:
            h = layer(h)

        # sample
        mu_q = h[:, :self.enc_dims[-1]]
        logvar_q = h[:, self.enc_dims[-1]:]
        std_q = torch.exp(0.5 * logvar_q)

        epsilon = torch.zeros_like(std_q).normal_(mean=0, std=0.01)
        sampled_z = mu_q + self.training * epsilon * std_q

        output = sampled_z
        for layer in self.decoder:
            output = layer(output)

        if self.training:
            kl_loss = ((0.5 * (-logvar_q + torch.exp(logvar_q) + torch.pow(mu_q, 2) - 1)).sum(1)).mean()
            return output, kl_loss
        else:
            return output

    def get_embeddings(self):
        self.eval()
        with torch.no_grad():
            eval_input = torch.Tensor(self.train_matrix.toarray()).to(self.device)

            h = F.dropout(F.normalize(eval_input), p=self.dropout, training=self.training)
            for layer in self.encoder:
                h = layer(h)
            mu_q = h[:, :self.enc_dims[-1]]
        self.train()
        return mu_q.detach().cpu().numpy()

    def predict(self, user_ids):
        self.eval()
        batch_eval_pos = self.train_matrix[user_ids]
        with torch.no_grad():
            eval_input = torch.Tensor(batch_eval_pos.toarray()).to(self.device)
            eval_output = self.forward(eval_input).detach().cpu().numpy()
        self.train()
        return eval_output


class EnLFT:
    def __init__(self, args, train_df, train_like, test_like, state_dict):
        self.num_user = args.num_user
        self.num_item = args.num_item

        self.t = args.t
        self.args = args
        self.local_bs = args.local_bs
        self.local_ep = args.local_ep
        self.anneal = args.anneal
        self.lr = args.lr
        self.reg = args.reg

        self.data = args.data

        self.train_like = train_like
        self.test_like = test_like
        user_array = train_df['userId'].values.reshape(-1)
        item_array = train_df['itemId'].values.reshape(-1)
        self.train_mat = coo_matrix((np.ones(len(train_df)), (user_array, item_array)),
                                    shape=(args.num_user, args.num_item)).toarray()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpu = torch.device("cpu")

        self.model = VAE(args, train_df, self.device)

        self.model.load_state_dict(state_dict)

        self.neighbor_users_list, self.neighbor_size_list, self.sim_mat = self.neighbor_users()

        tmp_sim_mat = copy.copy(self.sim_mat)
        user_sim = np.sum(tmp_sim_mat, axis=1)
        user_dissim = np.zeros_like(user_sim)
        self.selected_users = []
        for i in range(self.args.select):
            if i == 0:
                user_score = user_sim / (self.num_user - i)
            else:
                user_score = user_sim / (self.num_user - i) - 1.5 * user_dissim / i

            u = np.argmax(user_score)
            self.selected_users.append(u)

            tmp_sim_mat[:, u] = 0
            tmp_sim_mat[u, :] = -9999
            user_sim = np.sum(tmp_sim_mat, axis=1)

            user_dissim += self.sim_mat[:, u]

        self.selected_users = np.array(self.selected_users)

    def jaccard(self):
        num_rating_per_user = np.sum(self.train_mat, axis=1, keepdims=True)
        numerator = np.matmul(self.train_mat, self.train_mat.T)
        denominator = num_rating_per_user + num_rating_per_user.T - numerator
        denominator[denominator == 0] = 1
        Jaccard_mat = numerator / denominator
        Jaccard_mat *= (1 - np.eye(self.train_mat.shape[0]))
        return Jaccard_mat

    def cosine(self, x):
        numerator = np.matmul(x, x.T)
        denominator = np.sum(x ** 2, axis=1, keepdims=True) ** 0.5
        cosine_mat = numerator / denominator / denominator.T
        return cosine_mat

    def neighbor_users(self):
        Jaccard_mat = self.jaccard()
        distribution_mat = np.zeros_like(self.train_mat)
        for u in range(self.num_user):
            sim = Jaccard_mat[u]

            sim_threshold = self.args.sim_threshold
            sim_users = np.where(sim > sim_threshold)[0]
            k = 10
            if len(sim_users) < k:
                sim_users = np.argpartition(sim, -k)[-k:]

            alpha = self.args.alpha
            dist = alpha * self.train_mat[u, :] + (1 - alpha) * np.mean(self.train_mat[sim_users], axis=0, keepdims=True)
            distribution_mat[u, :] = dist

        cosine_mat = self.cosine(distribution_mat)
        neighbor_users_list = []
        neighbor_size_list = []
        s = 0
        for u in range(self.num_user):
            sim = copy.copy(cosine_mat[u])

            user_idx = np.where(sim > self.t)[0]
            k = 10
            if len(user_idx) < k:
                user_idx = np.argpartition(sim, -k)[-k:]
            sim_users = user_idx

            s += len(sim_users)
            neighbor_users_list.append(sim_users)
            neighbor_size_list.append(len(sim_users))
        return neighbor_users_list, np.array(neighbor_size_list), cosine_mat

    def local_update(self, user_id):
        local_train_mat = self.train_mat[self.neighbor_users_list[user_id], :]
        num_user = local_train_mat.shape[0]

        local_model = copy.deepcopy(self.model)
        local_model.to(self.device)
        local_model.train()

        optimizer = torch.optim.Adam(local_model.parameters(), lr=self.lr, weight_decay=self.reg)

        epoch_loss = []
        epochs = self.local_ep
        for iter in range(epochs):
            num_batch = int((num_user - 1) / self.local_bs) + 1
            random_idx = np.random.permutation(num_user)
            batch_loss = []
            for i in range(num_batch):
                batch_idx = random_idx[(i * self.local_bs):min(num_user, (i + 1) * self.local_bs)]
                batch_matrix = local_train_mat[batch_idx, :]
                batch_matrix = torch.FloatTensor(batch_matrix).to(self.device)
                optimizer.zero_grad()
                output, kl_loss = local_model.forward(batch_matrix)
                ce_loss = -((F.log_softmax(output, 1) * batch_matrix).sum(1)).mean()
                loss = ce_loss + kl_loss * self.anneal
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(np.mean(batch_loss))

        local_model.eval()
        user_input = torch.FloatTensor(self.train_mat).to(self.device)
        Rec = local_model(user_input).detach().cpu().numpy()
        return Rec

    def run(self):
        # ======================== Evaluate ========================

        epoch_eval_start = time()
        Rec = np.zeros_like(self.train_mat)

        sim_mat = self.sim_mat[:, self.selected_users]
        sim_mat_tmp = copy.copy(sim_mat)
        sim_mat[sim_mat < self.t] = 0
        idx = np.where(np.sum(sim_mat, axis=1) == 0)[0]
        sim_mat[idx, :] = sim_mat_tmp[idx, :]

        for i in tqdm(range(self.args.select)):
            u = self.selected_users[i]
            sim = sim_mat[:, i].reshape((-1, 1))
            u_Rec = self.local_update(u)
            Rec += u_Rec * sim
        Rec = Rec / np.sum(sim_mat, axis=1, keepdims=True)

        utility.MP_test_model_all(Rec, test_like, train_like, n_workers=10)

        with open('./Data/' + self.data + '/Rec_EnLFT.npy', "wb") as f:
            np.save(f, Rec.astype(np.float16))
        print("Save the best model")
        print("@" * 100)

        print("Testing time : %.2fs" % (time() - epoch_eval_start))

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Local_meta_VAE_DCsim')
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--reg', type=float, default=0.01, help='regularization')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--anneal', type=float, default=0.2, help='anneal')
    parser.add_argument('--hidden', type=int, default=100, help='latent dimension')
    parser.add_argument('--sim_threshold', type=float, default=0.1, help='sim_threshold')
    parser.add_argument('--local_ep', type=int, default=30, help='training epochs during testing')
    parser.add_argument('--select', type=int, default=100, help='number of anhor users')
    parser.add_argument('--alpha', type=float, default=0.7, help='alpha')
    parser.add_argument('--t', type=float, default=0.2, help='t')
    parser.add_argument('--data', type=str, default='ML1M', help='path to eval in the downloaded folder')

    args = parser.parse_args()
    print(args)

    with open('./Data/' + args.data + '/info.pkl', 'rb') as f:
        info = pickle.load(f)
        args.num_user = info['num_user']
        args.num_item = info['num_item']

    train_df = pd.read_csv('./Data/' + args.data + '/train_df.csv')

    train_like = list(np.load('./Data/' + args.data + '/user_train_like.npy', allow_pickle=True))
    test_like = list(np.load('./Data/' + args.data + '/user_test_like.npy', allow_pickle=True))

    state_dict = torch.load('./Data/' + args.data + '/WL.model')

    model = EnLFT(args, train_df, train_like, test_like, state_dict)
    model.run()




