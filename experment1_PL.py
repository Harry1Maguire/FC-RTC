import os
import pcap_proc6
import pcap_proc4
import pickle
from process import *
from model import myDISTILLER

testname = 'iptas_drop20'
ipv = 4
target = 7
device = 'cuda'
experiment_times = 10

def DatasetGenerate(filename, ipversion=4):
    fpath = f'.\\{filename}\\'
    files = os.listdir(fpath)
    fdict = {files[i]: i for i in range(len(files))}
    streams = {}
    for i in range(len(files)):
        obj = files[i]
        streams[obj] = dict()
        objfiles = os.listdir(os.path.join(fpath, obj))
        for file in objfiles:
            # print(os.path.join(obj, file))
            if ipversion == 6:
                streams[obj].update(pcap_proc6.pcap_proc(os.path.join(fpath, obj, file)))
            else:
                streams[obj].update(pcap_proc4.pcap_proc(os.path.join(fpath, obj, file)))
        print(len(streams[obj]))
    with open(f"{filename}.pkl", "wb") as f:
        pickle.dump([streams, files, fdict], f)

def TestTask(filename, no):
    no = str(no)
    with open(f"{filename}.pkl", "rb") as f:
        [streams, files, fdict] = pickle.load(f)
    datasets = []
    for obj in files:
        datasets.append(list(stream_to_data(streams[obj], fdict[obj])))
    dataset = datasets[0].copy()
    for i in range(1, len(datasets)):
        dataset[0] = dataset[0] + datasets[i][0]
        dataset[1] = dataset[1] + datasets[i][1]
        dataset[2] = dataset[2] + datasets[i][2]
    dataset_all = dataset[0] + dataset[1] + dataset[2]
    Trained_Model = myDISTILLER(target).to(device)
    Trained_Model.load_state_dict(torch.load('./model/normal_iptas1iptas_'+ no +'.pth'))
    datatest = DataLoader(dataset_all, batch_size=200, shuffle=True)
    tloss = []
    tacc = []
    loss_fn = nn.CrossEntropyLoss()
    for hdr, pay, y in datatest:
        y = y.type(torch.LongTensor)
        if y.shape[0] == 1:
            continue
        hdr = hdr.to(device)
        pay = pay.to(device)
        y = y.to(device)
        y_, hdr, pay = Trained_Model(hdr, pay)
        loss = loss_fn(y_, y)
        y_ = torch.max(y_, -1)[1]
        tacc.append(y_.eq(y.data.view_as(y_)).long().cpu().sum() / y_.shape[0])
        tloss.append(loss.cpu().item())
    tloss = np.mean(tloss)
    tacc = np.mean(tacc)
    print([tloss, tacc])
    return [tloss, tacc]

if not os.path.exists(f'{testname}.pkl'):
    DatasetGenerate(testname, ipv)
for i in range(1, experiment_times+1):
    TestTask(testname, i)



