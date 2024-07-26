from conmodel_coldstart import *

class Client_unbanlanced_type:
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
                print(y_)
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
            tloss.append(loss.cpu().item())
        tloss = np.mean(tloss)
        tacc = np.mean(tacc)
        return [tloss, tacc]