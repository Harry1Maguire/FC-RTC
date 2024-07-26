import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from supconloss import SupConLoss
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Nb = 784
Np = 32
BATCH = 128
lr = 5e-4
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

class myDISTILLER_coldstart(nn.Module):
    def __init__(self, target=3):
        super(myDISTILLER_coldstart, self).__init__()
        self.SM_PAY = nn.Sequential(
            nn.Conv1d(1, 16, 25, 1),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(16, 32, 25, 1),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Dropout(0.2),
            nn.Linear(76, 128),
            nn.ReLU()
        )
        self.SM_HDR1 = nn.GRU(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        self.SM_HDR2 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.SR = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.TS_App = nn.Sequential(
            nn.Conv1d(36, 1, 1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, target)
        )

    def forward(self, hdr, pay):
        hdr = hdr.unsqueeze(1)
        _, hdr = self.SM_HDR1(hdr)
        hdr = hdr.transpose(0, 1)
        hdr = self.SM_HDR2(hdr)
        pay = pay.unsqueeze(1)
        pay = self.SM_PAY(pay)
        flow = torch.cat((pay, hdr), 1)
        flow = self.SR(flow)
        pred = self.TS_App(flow)
        pred = pred.squeeze()
        # pred = nn.functional.log_softmax(pred, dim=1)
        return pred, hdr, pay


class Client_coldstart:
    def __init__(
            self,
            client_id: int,
            sets: list,
            tarnum: int
    ):
        self.model = myDISTILLER_coldstart(tarnum).to(device)
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
                loss_class = loss_fn(y_, y)
                loss_constrast_hdr = constrastive_loss(hdr, y) * 0.01
                loss_constrast_pay = constrastive_loss(pay, y) * 0.01
                loss = loss_class + loss_constrast_hdr + loss_constrast_pay
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
        all_labels = []
        all_scores = []
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
            y_out = y.detach().cpu()
            y_pre_out = y_.detach().cpu().numpy()
            y_scores = y_pre_out[:, 1]
            all_labels.extend(y_out.numpy())
            all_scores.extend(y_scores)
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
'''''''''
        precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
        average_precision = average_precision_score(all_labels, all_scores)
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(recall, precision, color='b', lw=2, label='PR curve (area = %0.2f)' % average_precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.legend(loc="lower left")
        plt.show()

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
'''''''''

