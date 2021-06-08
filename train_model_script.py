#%% Get Dependencies
import numpy as np
import torch
import matplotlib.pyplot as plt
import copy

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from models import LayerCNN, LayerCNNpp
from functions import num_of_params, plot_training_graph
from functions import get_datasets, sample_batch, drop_layers
from functions import preprocess, layer_prior, SMtoOU, OUtoSM, load_net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available(): print("Warning: using CPU not GPU")

#%% Set Options
opts = {
    "losses": [],
    "epoch": 0,
#model options
    "ksize": 3,
    "layer_ksize": 3,
    "num_layers": 3,
    "num_scales": 4,
    "c": 32,
    "c_out": 32,
    "batch_size": 16,
    "batchnorm": False,
    "dropout": 0,
    "act": ["relu","relu"],
#dataset options
    "dataset": "hard9",
    "resize": (64,128),
#preprocessing options
    "warp_border": {"bool": True, "h_std": [1,3], "v_std": [1,1.5]},
    "forward_passes": {"max_passes": 5, "epoch_inc": 3},
    "rescale_0_1": True,
    "drop_layer_chance": 0.2,
}

#%% Get Dataset

train_dataset, vali_dataset, _, _, _ = get_datasets(opts,resize=opts["resize"],num_layers_vec=range(1,6))

tr_dl = [DataLoader(i, batch_size=opts["batch_size"], shuffle=True, drop_last=True) for i in train_dataset]
va_dl = [DataLoader(i, batch_size=opts["batch_size"], shuffle=True) for i in vali_dataset]

#%% Initialize Network
num_epochs = 20

if opts["num_scales"]>1:
    net = LayerCNNpp(opts)
else:
    net = LayerCNN(opts)

net.to(device)
optimizer = optim.Adam(net.parameters(),lr=0.005,weight_decay=0)
optimizer.zero_grad()
torch.cuda.empty_cache()


criterion = nn.CrossEntropyLoss(reduction='mean')

opts["epoch"] = 0
opts["losses"] = {"bce_tr": [],
                  "bce_va": [],
                  "div_tr": [],
                  "div_va": [],
                  "times": []}
start = torch.cuda.Event(enable_timing=True)
ends = [torch.cuda.Event(enable_timing=True) for _ in range(3)]

sm = torch.nn.Softmax(dim=2)

#%% TRAIN
for _ in range(num_epochs):
    start.record()
    for i in opts["losses"]: opts["losses"][i].append([])
    net.train()
    for layer_batches in zip(*tr_dl):
        for X, L, GT in layer_batches:
            optimizer.zero_grad()
            L, GT = drop_layers(L, GT, p=opts["drop_layer_chance"])
            X, L, GT = preprocess(X,L,GT,opts,device)
            
            with torch.no_grad():
                for _ in range(np.random.randint(\
 np.min((np.floor(opts["epoch"]/opts["forward_passes"]["epoch_inc"]),
         opts["forward_passes"]["max_passes"]))+1)):
                    L[:,:,:,:,1:] = OUtoSM(layer_prior(SMtoOU(sm(net(X,L)[:,:,:,:,1:])),opts))*2-1
                    
            loss = criterion(net(X,L)[:,:,:,:,1:].squeeze(1), GT[:,:,1:]) 
                
            opts['losses']["bce_tr"][-1].append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1e1)
            optimizer.step()
    ends[0].record()
    with torch.no_grad():
        net.eval()
        for k, layer_batches in enumerate(zip(*tr_dl)):
            if k == 5:
                break
            for X, L, GT in layer_batches:
                L_hat, _,loss, _, _  = sample_batch(net,device,(X,L,GT),opts,label_pct=0.1)
                opts["losses"]["div_tr"][-1].append(loss[0].mean().item())
                
        ends[1].record()
        for k, layer_batches in enumerate(zip(*va_dl)):
            for X, L, GT in layer_batches:
                if k < 5:
                    L_hat, _,loss, _, _ = sample_batch(net,device,(X,L,GT),opts,label_pct=0.1)
                    opts["losses"]["div_va"][-1].append(loss[0].mean().item())
                
                
                L, GT = drop_layers(L, GT, p=opts["drop_layer_chance"])
                X, L, GT = preprocess(X,L,GT,opts,device)
                for _ in range(np.random.randint(\
                    np.min((np.floor(opts["epoch"]/opts["forward_passes"]["epoch_inc"]),
                            opts["forward_passes"]["max_passes"]))+1)):
                    L[:,:,:,:,1:] = OUtoSM(layer_prior(SMtoOU(sm(net(X,L)[:,:,:,:,1:])),opts))*2-1
                

                loss = criterion(net(X,L)[:,:,:,:,1:].squeeze(1), GT[:,:,1:]) 

                opts["losses"]["bce_va"][-1].append(loss.item())

        ends[2].record()
    torch.cuda.synchronize()
    
    opts["losses"]["times"][-1] = [start.elapsed_time(ends[0])/1000/60,
     (start.elapsed_time(ends[1])-start.elapsed_time(ends[0]))/1000/60,
     (start.elapsed_time(ends[2])-start.elapsed_time(ends[1]))/1000/60]
    
    print('epoch=%d; bce_tr=%.4f; bce_va=%.4f; div_tr=%.4f; div_va=%.4f; t=[%.2f,%.2f,%.2f]' % 
          (opts["epoch"]+1, 
            np.mean(opts["losses"]["bce_tr"][-1]),
            np.mean(opts["losses"]["bce_va"][-1]), 
            np.mean(opts["losses"]["div_tr"][-1]),
            np.mean(opts["losses"]["div_va"][-1]),
            *opts["losses"]["times"][-1]))
    if np.mean(opts["losses"]["div_va"][opts["epoch"]])\
        <=np.array(opts["losses"]["div_va"]).mean(1).min(): best_net = copy.deepcopy(net)
              
    opts["epoch"] += 1

#%% SAVE MODEL
base_path = "/zhome/75/a/138421/Desktop/BachelorProject/"
name = "h9_div_bce_test"
torch.save(best_net.state_dict(), 
           base_path+"models/best_"+name+".pth")
torch.save(net.state_dict(), 
           base_path+"models/"+name+".pth")
np.save(base_path+"models/"+name+".npy", opts)   
#%% Plot training graph
plot_training_graph(opts["losses"],log_bool=False,ylims=[(0.03,0.07),(0,3)])
#%% Display number of parameters
n_param = num_of_params(net,full_print=False)
#%% LOAD MODEL
best_net, net, opts = load_net("def_plus_plus",device,
   base_path="C:/Users/jakob/Desktop/DTU/Bachelor Project/saves/08_06_2021/")



