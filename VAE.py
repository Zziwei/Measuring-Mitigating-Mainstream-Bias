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
from scipy.sparse import csr_matrix
import utility

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

np.random.seed(1)
torch.random.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)


class VAE(nn.Module):
    def __init__(self, args, train_df, train_like, test_like, vali_like, device):
        super(VAE, self).__init__()

        self.num_item = args.num_item
        self.num_user = args.num_user

        train_df = train_df
        user_array = train_df['userId'].values.reshape(-1)
        item_array = train_df['itemId'].values.reshape(-1)
        self.train_matrix = csr_matrix((np.ones(len(train_df)), (user_array, item_array)), shape=(self.num_user, self.num_item))

        self.train_like = train_like
        self.test_like = test_like
        self.vali_like = vali_like

        self.enc_dims = [self.num_item, args.hidden]
        self.dec_dims = self.enc_dims[::-1]
        self.dims = self.enc_dims + self.dec_dims[1:]

        self.data = args.data

        self.batch_size = args.bs
        self.epoch = args.epoch
        self.lr = args.lr  # learning rate
        self.reg = args.reg  # regularization term trade-off
        self.dropout = args.dropout
        self.anneal = args.anneal

        self.display = args.display

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

    def train_model(self):

        best_result = 0.
        best_epoch = -1

        for epoch in range(1, self.epoch + 1):
            if epoch - best_epoch > 10:
                break

            self.train()

            # ======================== Train ========================
            epoch_loss = 0.0
            num_batch = int(self.num_user / self.batch_size) + 1
            random_idx = np.random.permutation(self.num_user)
            epoch_train_start = time()
            for i in tqdm(range(num_batch)):
                batch_idx = random_idx[(i * self.batch_size):min(self.num_user, (i + 1) * self.batch_size)]
                batch_matrix = self.train_matrix[batch_idx].toarray()
                batch_matrix = torch.FloatTensor(batch_matrix).to(self.device)

                batch_loss = self.train_model_per_batch(batch_matrix)
                epoch_loss += batch_loss

            epoch_train_time = time() - epoch_train_start
            print("Training //", "Epoch %d //" % epoch, " Total loss = {:.5f}".format(epoch_loss),
                  " Total training time = {:.2f}s".format(epoch_train_time))
            # ======================== Evaluate ========================
            if epoch % self.display == 0:
                self.eval()
                epoch_eval_start = time()
                Rec = self.predict_all()
                precision, recall, f_score, ndcg = utility.MP_test_model_all(Rec, self.vali_like, self.train_like,
                                                                             n_workers=10)
                utility.MP_test_model_all(Rec, self.test_like, self.train_like, n_workers=10)
                if np.mean(ndcg) > best_result:
                    best_epoch = epoch
                    best_result = np.mean(ndcg)

                    torch.save(self.state_dict(), './Data/' + self.data + '/VAE.model')
                    with open('./Data/' + self.data + '/Rec_VAE.npy', "wb") as f:
                        np.save(f, Rec.astype(np.float16))

                    print("Save the best model")
                    print("@" * 100)

                print("Testing time : %.2fs" % (time() - epoch_eval_start))

    def train_model_per_batch(self, batch_matrix):
        # zero grad
        self.optimizer.zero_grad()

        # model forwrad
        output, kl_loss = self.forward(batch_matrix)

        # loss
        ce_loss = -(F.log_softmax(output, 1) * batch_matrix).sum(1).mean()
        loss = ce_loss + kl_loss * self.anneal

        # backward
        loss.backward()

        # step
        self.optimizer.step()
        return loss

    def predict(self, user_ids):
        self.eval()
        batch_eval_pos = self.train_matrix[user_ids]
        with torch.no_grad():
            eval_input = torch.Tensor(batch_eval_pos.toarray()).to(self.device)
            eval_output = self.forward(eval_input).detach().cpu().numpy()
        self.train()
        return eval_output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--display', type=int, default=1, help='evaluate mode every X epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--reg', type=float, default=0.01, help='regularization')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--anneal', type=float, default=0.2, help='anneal')
    parser.add_argument('--hidden', type=int, default=100, help='latent dimension')
    parser.add_argument('--bs', type=int, default=512, help='batch size')
    parser.add_argument('--data', type=str, default='ML1M', help='path to eval in the downloaded folder')

    args = parser.parse_args()

    with open('./Data/' + args.data + '/info.pkl', 'rb') as f:
        info = pickle.load(f)
        args.num_user = info['num_user']
        args.num_item = info['num_item']

    train_df = pd.read_csv('./Data/' + args.data + '/train_df.csv')

    train_like = list(np.load('./Data/' + args.data + '/user_train_like.npy', allow_pickle=True))
    test_like = list(np.load('./Data/' + args.data + '/user_test_like.npy', allow_pickle=True))
    vali_like = list(np.load('./Data/' + args.data + '/user_vali_like.npy', allow_pickle=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('!' * 100)

    model = VAE(args, train_df, train_like, test_like, vali_like, device)
    model.train_model()



