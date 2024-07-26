import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from supconloss import SupConLoss
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Nb = 784
Np = 32
BATCH = 128
lr = 1e-3
TEMP_DIR = os.curdir
constrastive_loss = SupConLoss()


class myPreDataSet(object):
    def __init__(self, X, Y=None, Z=None):
        self.X = X
        self.Y = Y
        self.Z = Z
        if Y is None:
            self.mydata = X
        elif Z is None:
            self.mydata = [(x, y) for x, y in zip(X, Y)]
        else:
            self.mydata = [(x, y, z) for x, y, z in zip(X, Y, Z)]

    def __getitem__(self, idx):
        return self.mydata[idx]

    def __len__(self):
        return len(self.mydata)

    def __add__(self, other):
        return myPreDataSet(np.concatenate((self.X, other.X), axis=0),
                            np.concatenate((self.Y, other.Y), axis=0),
                            np.concatenate((self.Z, other.Z), axis=0))

    def part(self, perc: float):
        part_len = np.int(len(self.X) * perc)
        index = np.random.choice(len(self.X), part_len, replace=False)
        return myPreDataSet(self.X[index], self.Y[index], self.Z[index])

class traffic_CNN_RNN(nn.Module):
    def __init__(self, target=3):
        super(traffic_CNN_RNN, self).__init__()
        self.SM_HDR1 = nn.Sequential(
            nn.Conv1d(1, 32, 2, 1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 2, 1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=126*64, hidden_size=100, batch_first=True)
        self.TS_App2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(100, target)
        )

    def forward(self, hdr, pay):
        hdr = hdr.unsqueeze(1)
        hdr = self.SM_HDR1(hdr)
        batch_size, num_features, sequence_length = hdr.size()
        hdr1 = hdr.view(batch_size, sequence_length*num_features)
        hdr1, _ = self.lstm(hdr1)
        pred = self.TS_App2(hdr1)
        pred = pred.squeeze()
        # pred = nn.functional.log_softmax(pred, dim=1)
        return pred, hdr, pay


class Client_CNN_RNN:
    def __init__(
            self,
            client_id: int,
            sets: list,
            tarnum: int
    ):
        self.model = traffic_CNN_RNN(tarnum).to(device)
        self.id = client_id
        self.set = [DataLoader(sets[0], batch_size=BATCH, shuffle=True),
                    DataLoader(sets[1], batch_size=BATCH, shuffle=True),
                    DataLoader(sets[2], batch_size=BATCH, shuffle=True)]

    def train(self, epoch: int):
        loss_fn = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, )
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        for e in range(epoch):
            self.model.train()
            for hdr, pay, y in self.set[0]:
                y = y.type(torch.LongTensor)
                if y.shape[0] == 1:
                    continue
                hdr = hdr.to(device)
                pay = pay.to(device)
                y = y.to(device)
                y_, hdr, pay = self.model(hdr, pay)
                #print(y_)
                loss_class = loss_fn(y_, y)
                loss_constrast_hdr = constrastive_loss(hdr, y) * 0.01
                #loss_constrast_pay = constrastive_loss(pay, y) * 0.01
                loss = loss_class + loss_constrast_hdr
                opt.zero_grad()
                loss.backward()
                opt.step()
                if e == epoch-1:
                    y_ = torch.max(y_, -1)[1]
                    train_acc.append(y_.eq(y.data.view_as(y_)).long().cpu().sum() / y_.shape[0])
                    train_loss.append(loss.cpu().item())
            if e != epoch-1:
                continue
            train_loss = np.mean(train_loss)
            train_acc = np.mean(train_acc)
            self.model.eval()
            for hdr, pay, y in self.set[1]:
                y = y.type(torch.LongTensor)
                if y.shape[0] == 1:
                    continue
                hdr = hdr.to(device)
                pay = pay.to(device)
                y = y.to(device)
                y_, hdr, pay = self.model(hdr, pay)
                loss_class = loss_fn(y_, y)
                #loss_constrast_hdr = constrastive_loss(hdr, y) * 0.01
                #loss_constrast_pay = constrastive_loss(pay, y) * 0.01
                loss = loss_class #+ loss_constrast_hdr + loss_constrast_pay
                y_ = torch.max(y_, -1)[1]
                val_acc.append(y_.eq(y.data.view_as(y_)).long().cpu().sum() / y_.shape[0])
                val_loss.append(loss.cpu().item())
            val_loss = np.mean(val_loss)
            val_acc = np.mean(val_acc)
        return [train_loss, train_acc, val_loss, val_acc]

    def test(self):
        tloss = []
        tacc = []
        tpre = []
        f1_all = []
        loss_fn = nn.CrossEntropyLoss()
        self.model.eval()
        for hdr, pay, y in self.set[2]:
            y = y.type(torch.LongTensor)
            if y.shape[0] == 1:
                continue
            hdr = hdr.to(device)
            pay = pay.to(device)
            y = y.to(device)
            y_, hdr, pay = self.model(hdr, pay)
            loss = loss_fn(y_, y)
            y_ = torch.max(y_, -1)[1]
            tacc.append(y_.eq(y.data.view_as(y_)).long().cpu().sum() / y_.shape[0])
            p = precision_score(y.cpu().numpy(), y_.cpu().numpy(), average='weighted', zero_division=0)
            tpre.append(p)
            f1 = f1_score(y.cpu().numpy(), y_.cpu().numpy(), average='weighted', zero_division=0)
            f1_all.append(f1)
            tloss.append(loss.cpu().item())
        tloss = np.mean(tloss)
        tacc = np.mean(tacc)
        return [tloss, tacc, tpre, f1_all]
