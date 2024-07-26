import os
import pcap_proc6
import pcap_proc4
import pickle
from process import *
from model import myDISTILLER
from main_conmodel import load_dict
from conmodel_coldstart import myDISTILLER_coldstart

exp_time = 10
testname = 'iptas_drop5'
ipv = 4
target = 7
device = 'cuda'

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

def TestTask(filename, a):
    model_path = './exp-iptas1/model_iptas_' + str(a) + '.pth'
    hash_file = 'hash-exp-iptas1-' + str(a)
    with open(f"{filename}.pkl", "rb") as f:
        [streams, files, fdict] = pickle.load(f)
    datasets = []
    for obj in files:
        dict1, dict2, dict3 = load_dict(hash_file + '.txt')
        list1 = dict1[obj]
        list2 = dict2[obj]
        list3 = dict3[obj]
        datasets.append(list(stream_to_data_hashlist(streams[obj], fdict[obj], list1, list2, list3)))
    dataset = datasets[0].copy()
    for i in range(1, len(datasets)):
        dataset[0] = dataset[0] + datasets[i][0]
        dataset[1] = dataset[1] + datasets[i][1]
        dataset[2] = dataset[2] + datasets[i][2]
    dataset_test = dataset[2]
    Trained_Model = myDISTILLER_coldstart(target).to(device)
    Trained_Model.load_state_dict(torch.load(model_path))
    datatest = DataLoader(dataset_test, batch_size=200, shuffle=True)
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
    print(str(a), [tloss, tacc])
    return [tloss, tacc]

if not os.path.exists(f'{testname}.pkl'):
    DatasetGenerate(testname, ipv)
for i in range(1, exp_time+1):
    TestTask(testname, i)



