import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from dataset import Layers_Dataset
from models import LayerCNN, LayerCNNpp
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import DataLoader
from sklearn.metrics import adjusted_rand_score


def sample_batch(net,device,batch,opts,label_pct=0.1,orig_bool=True):
    with torch.no_grad():
        X, L, GT = batch
        X, L, GT = (X.clone().detach(),L.clone().detach(),GT.clone().detach())
        L_gt = GT.clone().detach() if orig_bool else L.clone().detach()
        X, L_gt, GT = preprocess(X,L_gt,GT,opts,device)
        
        w = X.size()[3]
        if isinstance(label_pct,int):
            w_include = label_pct
        else:
            w_include = int(label_pct*w)
        
        L_hat = L_gt.clone().detach()
        L_hat[:,:,:,:,w_include:] = 0
        
        sm = torch.nn.Softmax(dim=2)
        for i in range(w_include,w):
            L_tmp = net(X,L_hat)
            L_hat[:,:,:,:,i] = OUtoSM(layer_prior(SMtoOU(sm(L_tmp[:,:,:,:,i]),dim=2),
                                              opts,
                                              single_bool=True),
                                              dim=2)*2-1
        
        L_hat = L_hat/2+0.5
        L_gt = L_gt/2+0.5
        
        s_gt = OUtoLC(SMtoOU(L_gt),dim=3)
        s_hat = OUtoLC(SMtoOU(L_hat),dim=3)
        loss = []
        loss.append((s_gt-s_hat).abs().mean((1,2,3)).cpu().detach().numpy())
        loss.append((s_gt-s_hat).abs().pow(2).mean((1,2,3)).cpu().detach().numpy())
        
        L_tmp = sm(L_tmp)
        #s_gt = s_gt.squeeze()
        #s_hat = s_hat.squeeze()

    return L_hat, L_tmp, loss, s_gt, s_hat

def LCtoOU(s,dims=[64,128]):
    if not isinstance(s,np.ndarray):    
        s = s.cpu().numpy()
    L = torch.zeros(1,1,s.shape[0],dims[0],dims[1])
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            L[0,0,i,np.ceil(s[i,j]).astype(int):,j] = 1
            L[0,0,i,np.floor(s[i,j]).astype(int),j] = torch.tensor(np.ceil(s[i,j])-s[i,j])
            if np.ceil(s[i,j])==s[i,j]: L[0,0,i,np.floor(s[i,j]).astype(int),j] = 1
    return L

def OUtoLC(L_hat,dim=1):
    if L_hat.min() < -0.1:
        L_hat = L_hat/2+0.5
    s = (1-L_hat).sum(dim)
    return s

def OUtoSM(L,dim=2):
    if isinstance(L,np.ndarray):
        L2 = torch.from_numpy(L).clone()
        np_flag = True
    else:
        L2 = L.clone()
        np_flag = False
    
    dims = [i for i in range(len(L2.shape))]
    dims[0] = dim
    dims[dim] = 0
    L2 = L2.permute(tuple(dims))
    
    L2 = torch.cat((torch.ones_like(L2[0].unsqueeze(0)),L2),dim=0)
    L2[:-1] -= L2[1:].clone()
    
    L2 = L2.permute(tuple(dims))
    
    if np_flag:
        L2 = L2.numpy()
    return L2

def SMtoOU(L,dim=2):
    if isinstance(L,np.ndarray):
        L2 = torch.from_numpy(L).clone()
        np_flag = True
    else:
        L2 = L.clone()
        np_flag = False
        
    L2 = L.clone()
    dims = [i for i in range(len(L2.shape))]
    dims[0] = dim
    dims[dim] = 0
    L2 = L2.permute(tuple(dims))
    
    L2 = 1-L2.cumsum(0)[:-1]
    
    L2 = L2.permute(tuple(dims))
    
    if np_flag:
        L2 = L2.numpy()
    return L2

def layer_prior(L,opts,single_bool=False):
    if isinstance(L,np.ndarray):
        L2 = np.copy(L)
    else:
        L2 = np.copy(L.cpu())
    
    if L.min() < -0.1:
        L2 = L2/2+0.5
    L2 = np.expand_dims(L2,tuple(np.arange(5-len(L.shape))))

    if not single_bool: L2 = L2.transpose((0,1,2,4,3))

    L2 = np.concatenate((np.zeros((*L2.shape[:-1],1)),L2,np.ones((*L2.shape[:-1],1))),4)
    
    cost_image = L2.cumsum(4)+np.flip(np.flip(np.abs(L2-1),4).cumsum(4)-np.flip(np.abs(L2-1),4),4)-L2

    best_idx = cost_image.argmin(4)

    for d0 in range(L2.shape[0]):
        for d1 in range(L2.shape[1]):
            for d2 in range(L2.shape[2]):
                for d3 in range(L2.shape[3]):
                    idx = best_idx[d0,d1,d2,d3]
                    idx2 = idx+np.array([0 if idx==0 else -1,0 if idx==L2.shape[4]-1 else 1])
                    y = cost_image[d0,d1,d2,d3,idx2]-cost_image[d0,d1,d2,d3,idx]
                    delta = 0 if y[0]+y[1]<1e-14 else (y[0]-y[1])/(2*(y[0]+y[1]))
                    L2[d0,d1,d2,d3,:idx] = 0
                    L2[d0,d1,d2,d3,idx+1:] = 1
                    L2[d0,d1,d2,d3,idx] = 0.5-delta

    
    if single_bool: 
        L2 = L2[:,:,:,:,1:-1]
    else:
        L2 = L2[:,:,:,:,1:-1].transpose((0,1,2,4,3))
    
    L2 = torch.tensor(L2.reshape(L.shape))
    return L2

    
def preprocess(X,L,GT,opts,device):

    L = OUtoSM(layer_prior(L,opts))*2-1
    GT = (layer_prior(GT,opts)).sum(2).squeeze(1)
    GT = GT.floor()+(GT%1).bernoulli()

    if opts["rescale_0_1"]:
        X = (X-X.min())/(X.max()-X.min())*2-1
    else:
        X = X*2-1

    return X.to(device,dtype=torch.float),\
        L.to(device,dtype=torch.float), \
            GT.to(device,dtype=torch.long)

def num_of_params(net,full_print=False):
    n_param = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            n_param += param.data.numel()
            if full_print:
                print(name+", shape="+str(param.data.shape))
    print("Net has " + str(n_param) + " params.")
    return n_param

def plot_training_graph(losses,log_bool=False,ylims=None):
    fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(8,6))
    
    bce_tr = np.array(losses["bce_tr"])
    bce_va = np.array(losses["bce_va"])
    div_tr = np.array(losses["div_tr"])
    div_va = np.array(losses["div_va"])
    epochs = bce_tr.shape[0]
    
    bce_avg_tr = gaussian_filter(bce_tr.flatten(),sigma=bce_tr.size/epochs*0.01)
    #bce_avg_va = gaussian_filter(bce_va.flatten(),sigma=bce_va.size/epochs*0.01)
    ax[0].plot(np.linspace(0,epochs,bce_tr.size),bce_avg_tr,'-',alpha=0.4,color='navy',label='Moving average train loss')
    #ax[0].plot(np.linspace(0,epochs,bce_va.size),bce_avg_va,'-',alpha=0.4,color='red')
    ax[0].plot(np.linspace(1,epochs,epochs),bce_tr.mean(1),'-o',color='navy',fillstyle='none',label='Train loss')
    ax[0].plot(np.linspace(1,epochs,epochs),bce_va.mean(1),'-o',color='red',fillstyle='none',label='Vali. loss')
    
    ax[0].set_xlim([0,epochs])
    ax[0].set_ylim([0,None])
    ax[0].legend(loc='upper right')

    ax[1].plot(np.linspace(1,epochs,epochs),div_tr.mean(1),
             '-o',color='blue',fillstyle='none',alpha=0.7,label='Train sample error')
    ax[1].plot(np.linspace(1,epochs,epochs),div_va.mean(1),
         '-o',color='orange',fillstyle='none',alpha=0.7,label='Vali. sample error')

    ax[1].set_xlim([0,epochs])
    ax[1].set_ylim([0,None])
    ax[1].legend(loc='upper right')
    
    
    if not ylims is None:
        ax[0].set_ylim(ylims[0])
        ax[1].set_ylim(ylims[1])
        
    
    ax[0].set_ylabel("CE Loss")
    ax[0].set_xlabel("Epoch") 
    ax[1].set_ylabel("Mean L1 curve difference")
    ax[1].set_xlabel("Epoch")
    
    fig.tight_layout()
    plt.show()

def get_datasets(opts,resize=1,num_layers_vec=range(1,6),
                      base_path = "/zhome/75/a/138421/Desktop/BachelorProject/Data_Generation/data/",
                      tt_bool = True):
    train = []
    vali = []
    test = []
    real_va = Layers_Dataset(opts,num_layers=0,train_bool=False,
                          data_path=base_path+"real_validation",resize=resize)
    real_te = Layers_Dataset(opts,num_layers=0,train_bool=False,
                      data_path=base_path+"real_test",resize=resize)
    for i in num_layers_vec:
        train.append(Layers_Dataset(opts,num_layers=i,train_bool=tt_bool,
                                    data_path=base_path+opts["dataset"]+"/train",resize=resize))
        vali.append(Layers_Dataset(opts,num_layers=i,train_bool=False,
                                   data_path=base_path+opts["dataset"]+"/validation",resize=resize))
        test.append(Layers_Dataset(opts,num_layers=i,train_bool=False,
                                   data_path=base_path+opts["dataset"]+"/test",resize=resize))
    return train, vali, test, real_va, real_te

def drop_layers(L, GT, p):
    N_L = int(GT.shape[2])
    N_keep = int(1+(torch.rand(N_L-1)>p).sum())
    if p>0 and N_L>N_keep:
        for i in range(GT.shape[0]):
            idx, _ = torch.randperm(N_L)[0:N_keep].sort()
            L[i,:,0:N_keep] = L[i,:,idx]
            GT[i,:,0:N_keep] = GT[i,:,idx]
        return L[:,:,0:N_keep], GT[:,:,0:N_keep]
    else:
        return L, GT

def get_errors(nets,device,opts,idx_list=range(20),
                       size_list=None,
                       label_pct=0.1,
                       orig_bool=True,
                       num_batches=10,
                       N_layers=5,
                       batch_size=16):
    _, _, _, real_dataset64_va, real_dataset64_te = get_datasets(opts,resize=(64,128))
    _, _, _, real_dataset128_va, real_dataset128_te = get_datasets(opts,resize=(128,256))
    train_dataset, vali_dataset, test_dataset, _, _ = get_datasets(opts,resize=opts["resize"],
                                                                   tt_bool = False,
                                                                   num_layers_vec=range(1,N_layers+1))
    
    tr_dl = [DataLoader(i, batch_size=batch_size, shuffle=False) for i in train_dataset]
    va_dl = [DataLoader(i, batch_size=batch_size, shuffle=False) for i in vali_dataset]
    te_dl = [DataLoader(i, batch_size=batch_size, shuffle=False) for i in test_dataset]
    
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
    if not isinstance(nets,list):
        nets = [nets]
    #dls = [te_dl,va_dl,tr_dl]
    dls = [va_dl,te_dl,tr_dl]
    if size_list is None:
        size_list = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0]
    datasets = [real_dataset64_va, real_dataset128_va, real_dataset64_te, real_dataset128_te]
    LOSSES = [[[[[] for n in range(N_layers)] for k in range(4)] for j in range(5)] for i in range(len(nets))]
    S_hat = [[[[] for n in range(N_layers)] for j in range(5)] for i in range(len(nets))]
    S_gt = [[[] for n in range(N_layers)] for j in range(5)]
    for i in range(len(nets)):
        for j in range(2):
            for k in range(4):
                LOSSES[i][j][k] = []
                S_hat[i][j] = []
                S_gt[j] = []
    
    #Nets->dataset->Loss->n_layers
    with torch.no_grad():
        for net_num, net in enumerate(nets):
            net.eval()
            for real_idx in range(2):
                for idx in idx_list:
                    X,L,GT = datasets[size_list[idx]+real_idx*2][idx]
                    X = X.unsqueeze(0)
                    L = L.unsqueeze(0)
                    GT = GT.unsqueeze(0)
                    L_hat, L_tmp, loss, s_gt, s_hat = sample_batch(net,
                                  device,
                                  (X,L,GT),
                                  opts,
                                  label_pct=label_pct,
                                  orig_bool=orig_bool)
                    if net_num == 0:
                        for i in range(s_gt.shape[0]):
                            S_gt[real_idx].append(s_gt[i].cpu().numpy())

                    for i in range(s_hat.shape[0]):
                        S_hat[net_num][real_idx].append(s_hat[i].cpu().numpy())
                    
                    X, L, GT = preprocess(X,L,GT,opts,device)
                    
                    loss_bce = criterion(L_tmp[:,:,:,:,1:].squeeze(1), GT[:,:,1:]).mean((1,2))
                    
                    LOSSES[net_num][real_idx][0] += loss_bce.flatten().tolist()
                    LOSSES[net_num][real_idx][1] += list(loss[0].flatten())
                    LOSSES[net_num][real_idx][2] += list(loss[1].flatten())
                    for ARI_idx in range(GT.shape[0]):
                        LOSSES[net_num][real_idx][3].append(adjusted_rand_score(
                            L_hat[ARI_idx].argmax(1).flatten().cpu().numpy(),
                            GT[ARI_idx].flatten().cpu().numpy()))
                
            for dl_num, dl in enumerate(dls):
                for k, layer_batches in enumerate(zip(*dl)):
                    if k == num_batches:
                        break
                    for n_layers, batch in enumerate(layer_batches):
                        L_hat, L_tmp, loss, s_gt, s_hat = sample_batch(net,
                                                      device,
                                                      batch,
                                                      opts,
                                                      label_pct=label_pct,
                                                      orig_bool=orig_bool)
                        if net_num == 0:
                            for i in range(s_gt.shape[0]):
                                S_gt[dl_num+2][n_layers].append(s_gt[i].cpu().numpy())
        
                        for i in range(s_hat.shape[0]):
                            S_hat[net_num][dl_num+2][n_layers].append(s_hat[i].cpu().numpy())
                        
                        X, L, GT = batch
                        X, L, GT = preprocess(X,L,GT,opts,device)
                        
                        loss_bce = criterion(L_tmp[:,:,:,:,1:].squeeze(1), GT[:,:,1:]).mean((1,2))
                        LOSSES[net_num][dl_num+2][0][n_layers] += loss_bce.flatten().tolist()
                        LOSSES[net_num][dl_num+2][1][n_layers] += list(loss[0].flatten())
                        LOSSES[net_num][dl_num+2][2][n_layers] += list(loss[1].flatten())
                        for ARI_idx in range(GT.shape[0]):
                            LOSSES[net_num][dl_num+2][3][n_layers].append(adjusted_rand_score(
                                L_hat[ARI_idx].argmax(1).flatten().cpu().numpy(),
                                GT[ARI_idx].flatten().cpu().numpy()))
                            
    mean_losses = []
    for i in range(len(LOSSES)):
        mean_losses.append([])
        for j in range(len(LOSSES[0])):
            mean_losses[i].append([])
            for k in range(len(LOSSES[0][0])):
                mean_losses[i][j].append(np.array(LOSSES[i][j][k]).mean())
    mean_losses = np.array(mean_losses)
    
    flattened_losses = []
    for i in range(len(LOSSES)):
        flattened_losses.append([])
        for j in range(len(LOSSES[0])):
            flattened_losses[i].append([])
            for k in range(len(LOSSES[0][0])):
                flattened_losses[i][j].append(np.array(LOSSES[i][j][k]).flatten())
    return {"LOSSES": LOSSES,
              "mean_losses": mean_losses,
              "flattened_losses": flattened_losses,
              "S_gt": S_gt,
              "S_hat": S_hat}

def load_net(name,device,base_path="/zhome/75/a/138421/Desktop/BachelorProject/",best_bool=False):
    opts = np.load("models/"+name+".npy",allow_pickle='TRUE').item()
    checkpoint = torch.load(base_path+"models/"+name+".pth",
                                               map_location=device)
    if best_bool: checkpoint_best = torch.load(base_path+"models/"+name+"_best.pth",
                                               map_location=device)
    if opts["num_scales"]>1:
        net = LayerCNNpp(opts)
        if best_bool: best_net = LayerCNNpp(opts)
    else:
        net = LayerCNN(opts)
        if best_bool: best_net = LayerCNN(opts)
    net.to(device)
    net.load_state_dict(checkpoint)
    if best_bool:
        best_net.to(device)
        best_net.load_state_dict(checkpoint_best)
    else: best_net = []
    return best_net, net, opts


def automatic_init(net,device,dataset,opts,idx=-1,selection=None,n_r=10,cluster_dist=1.5):
    if isinstance(dataset,tuple):
        if len(dataset)==3:
            X, L, GT = dataset
    else:        
        if isinstance(dataset,list):
            dataset = dataset[np.random.randint(len(dataset))]
        if idx < 0:
            idx = np.random.randint(len(dataset))
            
        X, L, GT = dataset[idx]
    X = X.clone().unsqueeze(0)
    
    X = torch.flip(X,(3,)).repeat(n_r,1,1,1)
        
    h = X.shape[2]
    w = X.shape[3]
    
    l_r = torch.linspace(h/(n_r+1),n_r*h/(n_r+1),n_r)
    s_r = l_r.repeat(w,1).T
    L_init = LCtoOU(s_r,dims=[h,w]).permute((2,1,0,3,4))
    
    L_hat, L_tmp, loss, _, s_hat = sample_batch(net,device,
                         (X,L_init,L_init),
                         opts,
                         label_pct=1,
                         orig_bool=True)
    
    X = torch.flip(X,(3,))
    L_hat = torch.flip(L_hat,(4,))
    L_tmp = torch.flip(L_tmp,(4,))
    s_hat = torch.flip(s_hat,(3,))
    cost = (L_hat[:,:,:,:,:w-1]-L_tmp[:,:,:,:,:w-1]).abs().mean((1,2,3,4)).cpu().numpy()
    cost += (((s_hat<1).squeeze().cpu().numpy().mean(1)+(s_hat>(h-1)).squeeze().cpu().numpy().mean(1))>0.2).astype(float)
    cluster = AgglomerativeClustering(n_clusters=None,
                          distance_threshold=cluster_dist, 
                          affinity='euclidean', 
                          linkage='single')
    
    l_left = s_hat.squeeze()[:,0].cpu().numpy().reshape(-1, 1)
    clusters = cluster.fit_predict(l_left)
    clusters_cost = []
    clusters_unique = np.unique(clusters)
    for i in clusters_unique:
        clusters_cost.append(cost[clusters==i].min())
    
    clusters_cost = np.array(clusters_cost)
    
    means = np.array([np.mean(l_left[clusters==i]) for i in clusters_unique])
    order = np.argsort(means)
    means = means[order]
    clusters_cost = clusters_cost[order]
    
    clusters_order = np.argsort(clusters_cost)
    if isinstance(selection,int):
        used_clusters = np.ones_like(clusters_cost)<0
        used_clusters[clusters_order[:selection]] = True
    elif isinstance(selection,float):
        used_clusters = clusters_cost<selection
        if used_clusters.sum() < 0.5:
            used_clusters[clusters_order[0]] = True
    else:
        used_clusters = np.ones_like(clusters_cost)>0 
    
    init_vals = torch.tensor(means[used_clusters])
    s_l = init_vals.repeat(w,1).T
    L_init = LCtoOU(s_l,dims=[h,w])
    s_hat_init = s_hat
    return L_init, means, clusters, cost, clusters_cost, s_hat_init

def get_errors_auto_init(nets,
                         device,
                         opts,
                         idx_list=range(20),
                         size_list=None,
                         label_pct=1,
                         num_elements=3,
                         N_layers=5,
                         selection=None,
                         n_r=10,
                         cluster_dist=1.5):

    _, _, _, real_dataset64_va, real_dataset64_te = get_datasets(opts,resize=(64,128))
    _, _, _, real_dataset128_va, real_dataset128_te = get_datasets(opts,resize=(128,256))
    train_dataset, vali_dataset, test_dataset, _, _ = get_datasets(opts,resize=opts["resize"],
                                                                   tt_bool = False,
                                                                   num_layers_vec=range(1,N_layers+1))
    syn_datasets = [test_dataset,vali_dataset,train_dataset]
    
    if not isinstance(nets,list):
        nets = [nets]
        
    if size_list is None:
        size_list = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0]
    datasets = [real_dataset64_va, real_dataset128_va, real_dataset64_te, real_dataset128_te]
    LOSSES = [[[[] for n in range(N_layers)] for j in range(5)] for i in range(len(nets))]
    S_hat = [[[[] for n in range(N_layers)] for j in range(5)] for i in range(len(nets))]
    S_gt = [[[] for n in range(N_layers)] for j in range(5)]
    init_info = [[[[] for n in range(N_layers)] for j in range(5)] for i in range(len(nets))]
    
    for i in range(len(nets)):
        for j in range(2):
            LOSSES[i][j] = []
            S_hat[i][j] = []
            init_info[i][j] = [] 
            S_gt[j] = []
            
    
    #Nets->dataset->n_layers
    with torch.no_grad():
        for net_num, net in enumerate(nets):
            print(net_num)
            net.eval()
            for real_idx in range(2):
                for idx in idx_list:
                    X,L,GT = datasets[size_list[idx]+real_idx*2][idx]
                    X = X.unsqueeze(0)
                    L = L.unsqueeze(0)
                    GT = GT.unsqueeze(0)
                    L_init, means, clusters, cost, clusters_cost, s_hat_init = automatic_init(net,
                                        device,
                                        datasets[size_list[idx]+real_idx*2],
                                        opts,
                                        idx=idx,
                                        selection=GT.shape[2] if selection==0 else selection,
                                        n_r=n_r,
                                        cluster_dist=cluster_dist)
                    
                    L_hat, L_tmp, _, _, s_hat = sample_batch(net,
                                        device,
                                        (X,L_init,L_init),
                                        opts,
                                        label_pct=label_pct,
                                        orig_bool=False)
                    
                    if net_num == 0:
                        s_gt = OUtoLC(GT,dim=3).squeeze()
                        S_gt[real_idx].append(s_gt)
    
                    S_hat[net_num][real_idx].append(s_hat.squeeze().cpu().numpy())
                    
                    _, _, GT = preprocess(X,L,GT,opts,device)
                    LOSSES[net_num][real_idx].append(adjusted_rand_score(
                        L_hat.argmax(2).flatten().cpu().numpy(),
                        GT.flatten().cpu().numpy()))
                    
                    init_info[net_num][real_idx].append({"means": means,
                                        "clusters": clusters,
                                        "cost": cost,
                                        "clusters_cost": clusters_cost,
                                        "s_hat_init": s_hat_init.squeeze().cpu().numpy()})
            
            for syn_num, dataset_list in enumerate(syn_datasets):
                for n_layers, dataset in enumerate(dataset_list):
                    for idx in range(num_elements):
                        X,L,GT = dataset[idx]
                        X = X.unsqueeze(0)
                        L = L.unsqueeze(0)
                        GT = GT.unsqueeze(0)
                        
                        L_init, means, clusters, cost, clusters_cost, s_hat_init = automatic_init(net,
                                                              device,
                                                              dataset,
                                                              opts,
                                                              idx=idx,
                                                              selection=GT.shape[2] if selection==0 else selection,
                                                              n_r=n_r,
                                                              cluster_dist=cluster_dist)
                        
                        L_hat, L_tmp, _, _, s_hat = sample_batch(net,
                                      device,
                                      (X,L_init,L_init),
                                      opts,
                                      label_pct=label_pct,
                                      orig_bool=False)
                        
                        if net_num == 0:
                            s_gt = OUtoLC(GT,dim=3).squeeze()
                            S_gt[syn_num+2][n_layers].append(s_gt)
        
                        S_hat[net_num][syn_num+2][n_layers].append(s_hat.squeeze().cpu().numpy())
                            
                        
                        _, _, GT = preprocess(X,L,GT,opts,device)
                        LOSSES[net_num][syn_num+2][n_layers].append(adjusted_rand_score(
                            L_hat.argmax(2).flatten().cpu().numpy(),
                            GT.flatten().cpu().numpy()))
                        
                        init_info[net_num][syn_num+2][n_layers].append({"means": means,
                                            "clusters": clusters,
                                            "cost": cost,
                                            "clusters_cost": clusters_cost,
                                            "s_hat_init": s_hat_init.squeeze().cpu().numpy()})
                        
    mean_losses = []
    for i in range(len(LOSSES)):
        mean_losses.append([])
        for j in range(len(LOSSES[0])):
            mean_losses[i].append(np.array(LOSSES[i][j]).mean())
    mean_losses = np.array(mean_losses)
    
    flattened_losses = []
    for i in range(len(LOSSES)):
        flattened_losses.append([])
        for j in range(len(LOSSES[0])):
            flattened_losses[i].append(np.array(LOSSES[i][j]).flatten())
    
    return {"LOSSES": LOSSES,
            "mean_losses": mean_losses,
            "flattened_losses": flattened_losses,
            "S_gt": S_gt,
            "S_hat": S_hat,
            "init_info": init_info}

def plot_results(losses,device,opts,
                 idx=None,
                 model_names=None,
                 dataset_idx=None,
                 net_idx=None,
                 n_layers_idx=None,
                 random_idx=True,
                 size_list=None):
    
    N_datasets = len(losses["S_hat"][0])
    N_nets = len(losses["S_hat"])
    N_layers = len(losses["S_hat"][0][2])
    N_syn = len(losses["S_hat"][0][2][0])
    N_real = len(losses["S_hat"][0][0])
    
    
    if isinstance(idx,int):
        N = idx
        if idx==0:
            idx = [idx for _ in range(N)]
    elif isinstance(idx,list) or isinstance(idx,range):
        if isinstance(idx,range):
            idx = list(idx)
        N = len(idx)
    elif idx is None:
        N = 5
    
        
    if isinstance(dataset_idx,int):
        dataset_idx = [dataset_idx for _ in range(N)]
    elif dataset_idx is None:
        if random_idx:
            dataset_idx = [np.random.randint(N_datasets) for _ in range(N)]
        else:
            dataset_idx = [0 for _ in range(N)]
        
    if isinstance(net_idx,int):
        N2 = net_idx
        net_idx = [i for i in range(N2)]
    if isinstance(net_idx,list):
        N2 = len(net_idx)
    elif net_idx is None:
        N2 = 3
        if random_idx:
            net_idx = [np.random.randint(N_nets) for _ in range(N2)]
        else:
            net_idx = [i for i in range(N2)]
    
    if isinstance(n_layers_idx,int):
        n_layers_idx = [n_layers_idx for _ in range(N)]
    elif n_layers_idx is None:
        if random_idx:
            n_layers_idx = [np.random.randint(N_layers) for _ in range(N)]
        else:
            n_layers_idx = [i%N_layers for i in range(N)]
        
    if idx is None or isinstance(idx,int):
        if random_idx:
            idx = []
            for i in range(N):
                if dataset_idx[i]<2:
                    idx.append(np.random.randint(N_real))                
                else:
                    idx.append(np.random.randint(N_syn))
        else:
            if max(n_layers_idx)-min(n_layers_idx)>0.5:
                idx = [0 for _ in range(N)]
            else:
                idx = [i for i in range(N)]
                
                
    if model_names is None:
        model_names = [str(net_idx[i]) for i in range(N2)]
                
    if size_list is None:
        size_list = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0]
        
    fig,axs = plt.subplots(nrows=N,ncols=N2+1,figsize=(5*(N2+1),2.5*N))
    #Nets->dataset->Loss->n_layers
    #return LOSSES, mean_losses, flattened_losses, S_gt, S_hat, init_info
    
    _, _, _, real_dataset64_va, real_dataset64_te = get_datasets(opts,resize=(64,128))
    _, _, _, real_dataset128_va, real_dataset128_te = get_datasets(opts,resize=(128,256))
    train_dataset, vali_dataset, test_dataset, _, _ = get_datasets(opts,resize=opts["resize"],
                                                                   tt_bool = False,
                                                                   num_layers_vec=range(1,N_layers+1))
    syn_datasets = [test_dataset,vali_dataset,train_dataset]
    
    real_names = ["Bone#"+str(i) for i in range(1,9)]+\
                 ["Retina#"+str(i) for i in range(1,9)]+\
                 ["Wood#1","Wood#2","Ferrero#1","Ferrero#2"]
    real_dataset_names = ["Vali. ","Test "]
    syn_dataset_names = ["Test ", "Vali. ","Train "]
    datasets = [real_dataset64_va, real_dataset128_va, real_dataset64_te, real_dataset128_te]
    
    print(N,N2,dataset_idx,net_idx,n_layers_idx,idx)
    for i in range(N):
        if i==0:
            axs[0,0].set_title("Ground Truth")
        if dataset_idx[i] < 2:
            X,_,_ = datasets[size_list[idx[i]]+dataset_idx[i]*2][idx[i]]
            X = X.squeeze().detach().cpu().numpy()
            axs[i,0].imshow(X,cmap='gray')
            axs[i,0].plot(losses["S_gt"][dataset_idx[i]][idx[i]].T)
        else:
            X,_,_ = syn_datasets[dataset_idx[i]-2][n_layers_idx[i]][idx[i]]
            X = X.squeeze().detach().cpu().numpy()
            axs[i,0].imshow(X,cmap='gray')
            axs[i,0].plot(losses["S_gt"][dataset_idx[i]][n_layers_idx[i]][idx[i]].T) 
        h = X.shape[0]
        w = X.shape[1]
        axs[i,0].set_ylim([h-1,0])
        axs[i,0].set_xlim([0,w-1])
        axs[i,0].set_xticks([]),axs[i,0].set_yticks([])
        axs[i,0].set_xticks([]),axs[i,0].set_yticks([])
        if dataset_idx[i] < 2:
            axs[i,0].set_ylabel(real_dataset_names[dataset_idx[i]]+real_names[idx[i]])
        else:
            axs[i,0].set_ylabel(syn_dataset_names[dataset_idx[i]-2]+" im#"+str(idx[i]+1)+", n="+str(1+n_layers_idx[i]))
        for j in range(N2):
            if i==0:
                axs[0,j+1].set_title("Model="+model_names[j])
            if dataset_idx[i] < 2:
                axs[i,j+1].imshow(X,cmap='gray')
                axs[i,j+1].plot(losses["S_hat"][net_idx[j]][dataset_idx[i]][idx[i]].T)
                axs[i,j+1].set_title("L1="+str(round(losses["LOSSES"][net_idx[j]][dataset_idx[i]][1][idx[i]],3))+\
                                    ", L2="+str(round(losses["LOSSES"][net_idx[j]][dataset_idx[i]][2][idx[i]],3))+\
                                    ", ARI="+str(round(losses["LOSSES"][net_idx[j]][dataset_idx[i]][3][idx[i]],3)))
            else:
                axs[i,j+1].imshow(X,cmap='gray')
                axs[i,j+1].plot(losses["S_hat"][net_idx[j]][dataset_idx[i]][n_layers_idx[i]][idx[i]].T)
                axs[i,j+1].set_title("L1="+str(round(losses["LOSSES"][net_idx[j]][dataset_idx[i]][1][n_layers_idx[i]][idx[i]],3))+\
                    ", L2="+str(round(losses["LOSSES"][net_idx[j]][dataset_idx[i]][2][n_layers_idx[i]][idx[i]],3))+\
                    ", ARI="+str(round(losses["LOSSES"][net_idx[j]][dataset_idx[i]][3][n_layers_idx[i]][idx[i]],3)))
            #Nets->dataset->Loss->n_layers
            axs[i,j+1].set_ylim([h-1,0])
            axs[i,j+1].set_xlim([0,w-1])
            axs[i,j+1].set_xticks([]),axs[i,j+1].set_yticks([])
            axs[i,j+1].set_xticks([]),axs[i,j+1].set_yticks([])
            
    """plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.8, 
                        top=0.8, 
                        wspace=0.1, 
                        hspace=0.1)"""

def plot_training_graph_multi(losses_list,names,ylims=None):
    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(12,4))
    ax[0].set_title("Training loss")
    ax[1].set_title("Validation sampling error")
    for i, losses in enumerate(losses_list):
            
        bce_tr = np.array(losses["bce_tr"])
        #bce_va = np.array(losses["bce_va"])
        #div_tr = np.array(losses["div_tr"])
        div_va = np.array(losses["div_va"])
        epochs = bce_tr.shape[0]

        ax[0].plot(np.linspace(1,epochs,epochs),bce_tr.mean(1),fillstyle='none',alpha=0.7,label=names[i])
        ax[1].plot(np.linspace(1,epochs,epochs),div_va.mean(1),fillstyle='none',alpha=0.7,label=names[i])
        
    ax[0].set_xlim([0,epochs])
    ax[0].set_ylim([0,None])
    ax[0].legend(loc='upper right')

    ax[1].set_xlim([0,epochs])
    ax[1].set_ylim([0,None])
    ax[1].legend(loc='upper right')
        
        
    if not ylims is None:
        ax[0].set_ylim(ylims[0])
        ax[1].set_ylim(ylims[1])
            
        
    ax[0].set_ylabel("CE Loss")
    ax[0].set_xlabel("Epoch")
    ax[1].set_ylabel("L1 error")
    ax[1].set_xlabel("Epoch")
    
    fig.tight_layout()

