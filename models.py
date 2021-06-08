import torch
import torch.nn.functional as F
import torch.nn as nn
    

class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x_tanh, x_sigmoid = torch.chunk(x,2,dim=1)
        return torch.tanh(x_tanh)*torch.sigmoid(x_sigmoid)


class ConvBlock(nn.Module):
    def __init__(self, opts, type_str='B', c=32):
        super(ConvBlock, self).__init__()
        
        self.opts = opts
        self.k_size = opts["ksize"]
        if torch.tensor(self.k_size).numel() == 1:
            self.k_size = [self.k_size,self.k_size]
        self.type_str = type_str

        self.act0 = GatedActivation() if opts["act"][0]=='gated' else F.relu
        self.act1 = GatedActivation() if opts["act"][1]=='gated' else F.relu
        self.act0_mult = 2 if opts["act"][0]=='gated' else 1
        self.act1_mult = 2 if opts["act"][1]=='gated' else 1
        
        self.XConv0 = nn.Conv2d(in_channels=(c if type_str=='B' else 1),
                                out_channels=c*self.act0_mult,
                                kernel_size=self.k_size[0],
                                padding=self.k_size[0]//2)
        self.XConv1 = nn.Conv2d(in_channels=c,
                                out_channels=c*self.act1_mult,
                                kernel_size=self.k_size[1],
                                padding=self.k_size[1]//2)
        
        self.XtoLConv = nn.Conv2d(in_channels=c,
                                  out_channels=c*self.act0_mult,
                                  kernel_size=1,
                                  padding=0)
        

        self.LConv0 = nn.Conv3d(in_channels=(c if type_str=='B' else 1),
            out_channels=c*self.act0_mult,
            kernel_size=(1,self.k_size[0],self.k_size[0]//2+(type_str=='B')),
            padding=(0,self.k_size[0]//2,self.k_size[0]//2))
        self.LConv1 = nn.Conv3d(in_channels=c,
            out_channels=c*self.act1_mult,
            kernel_size=(opts["layer_ksize"],self.k_size[1],self.k_size[1]//2+1),
            padding=(opts["layer_ksize"]//2,self.k_size[1]//2,self.k_size[1]//2))
                
        if self.opts["batchnorm"]:
            self.BN_X0 = nn.BatchNorm2d(c*self.act0_mult)
            self.BN_X1 = nn.BatchNorm2d(c)
            self.BN_L0 = nn.BatchNorm3d(c*self.act0_mult)
            self.BN_L1 = nn.BatchNorm3d(c)
        else:
            self.BN_X0 = nn.Identity()
            self.BN_X1 = nn.Identity()
            self.BN_L0 = nn.Identity()
            self.BN_L1 = nn.Identity()
        
        self.drop = nn.Dropout(self.opts["dropout"]) if opts["dropout"]>0 else nn.Identity()
        
    def forward(self, x):
        
        X_in,L_in,skip = x[0], x[1], x[2]
        
        X = self.act0(self.BN_X0(self.XConv0(X_in)))
        X = self.act1(self.XConv1(X))
        X = self.BN_X1(X + self.drop(X_in)) if self.type_str=='B' else self.BN_X1(X)
        
        XtoL = self.XtoLConv(X)
        
        L = self.LConv0(L_in)[:,:,:,:,:-(self.k_size[0]//2)-(self.type_str=='A')]
        L = self.LConv1(self.act0(self.BN_L0(L)+XtoL.unsqueeze(2)))
        L = self.act1(L)[:,:,:,:,:-(self.k_size[1]//2)]
        L = self.BN_L1(L + self.drop(L_in)) if self.type_str=='B' else self.BN_L1(L)
        
        skip = L if self.type_str=='A' else skip + L
        return {0: X, 1: L, 2: skip}
    
    
class LayerCNN(nn.Module):
    def __init__(self, opts):
        super(LayerCNN, self).__init__()
        
        self.opts = opts
        self.FirstBlock = ConvBlock(self.opts, type_str='A',c=opts["c"])

        self.StackedBlocks = nn.Sequential(
*[ConvBlock(opts, type_str='B',c=opts["c"]) for _ in range(opts["num_layers"])]
        )
        self.LastConvHidden = nn.Conv3d(in_channels=opts["c"],out_channels=opts["c_out"],kernel_size=1)

        self.LastConv = nn.Conv3d(in_channels=self.opts["c_out"],out_channels=1,kernel_size=1)

    def forward(self, X, L):        
        X, L, skip = self.FirstBlock({0: X, 
                                      1: L, 
                                      2: []
                                      }).values()

        _, _, skip = self.StackedBlocks({0: X, 1: L, 2: skip}).values()

        L = F.relu(skip)
        L = F.relu(self.LastConvHidden(L))
        L = self.LastConv(L)
        return L

class DownScale(nn.Module):
    def __init__(self, c_in, c_out):
        super(DownScale, self).__init__()
        self.DownX = nn.Conv2d(in_channels=c_in,
                               out_channels=c_out,
                               kernel_size=2,
                               stride=2,
                               padding=0)
        self.DownL = nn.Conv3d(in_channels=c_in,
                               out_channels=c_out,
                               kernel_size=(1,2,2),
                               stride=(1,2,2),
                               padding=(0,0,1))

    def forward(self, X, L):
        X = self.DownX(X)
        L = self.DownL(L)[:,:,:,:,:-1]
        return X, L
    
class UpScale(nn.Module):
    def __init__(self, c_in, c_out):
        super(UpScale, self).__init__()
        self.UpX = nn.ConvTranspose2d(in_channels=c_in,
                               out_channels=c_out,
                               kernel_size=2,
                               stride=2)
        self.UpL = nn.ConvTranspose3d(in_channels=c_in,
                               out_channels=c_out,
                               kernel_size=(1,2,2),
                               stride=(1,2,2))
        self.Upskip = nn.ConvTranspose3d(in_channels=c_in,
                               out_channels=c_out,
                               kernel_size=(1,2,2),
                               stride=(1,2,2))

    def forward(self, X, L, skip):
        X = self.UpX(X)
        L = self.UpL(L)
        skip = self.Upskip(skip)
        return X, L, skip

class LayerCNNpp(nn.Module):
    def __init__(self, opts):
        super(LayerCNNpp, self).__init__()
        
        self.opts = opts
        if torch.tensor(self.opts["c"]).numel() == 1:
            self.opts["c"] = [self.opts["c"]]*self.opts["num_scales"]
        
        self.FirstBlock = ConvBlock(self.opts, type_str='A',c=opts["c"][0])
        
        self.DownBlocks = torch.nn.ModuleList([nn.Sequential(
*[ConvBlock(opts, type_str='B',c=self.opts["c"][i]) for _ in range(opts["num_layers"])]
        ) for i in range(opts["num_scales"])])
        
        self.UpBlocks = torch.nn.ModuleList([nn.Sequential(
*[ConvBlock(opts, type_str='B',c=self.opts["c"][i]) for _ in range(opts["num_layers"])]
        ) for i in range(opts["num_scales"])])
        
        self.DownScales = torch.nn.ModuleList(
[DownScale(self.opts["c"][i],self.opts["c"][i+1]) for i in range(opts["num_scales"]-1)])
        
        self.UpScales = torch.nn.ModuleList(
[UpScale(self.opts["c"][i+1],self.opts["c"][i]) for i in range(opts["num_scales"]-1)])
        
        
        self.LastConvHidden = nn.Conv3d(in_channels=opts["c"][0],out_channels=opts["c_out"],kernel_size=1)
        self.LastConv = nn.Conv3d(in_channels=self.opts["c_out"],out_channels=1,kernel_size=1)

    def forward(self, X, L):        
        X, L, skip = self.FirstBlock({0: X, 
                                      1: L, 
                                      2: 0
                                      }).values()
        skip_list = []
        X_list = []
        L_list = []
        #ENCODER
        for i in range(self.opts["num_scales"]):
            X, L, skip = self.DownBlocks[i]({0: X, 1: L, 2: skip if i==0 else 0}).values()
            
            if i < self.opts["num_scales"]-1:
                X_list.append(X)
                L_list.append(L)
                skip_list.append(skip)
                X, L = self.DownScales[i](X,L)
        #DECODER
        for i in reversed(range(self.opts["num_scales"])):
            if i == self.opts["num_scales"]-1:
                X, L, skip = self.UpBlocks[i]({0: X,
                                               1: L, 
                                               2: skip}).values()
            else:
                X, L, skip = self.UpBlocks[i]({0: X_list.pop()+X,
                                               1: L_list.pop()+L, 
                                               2: skip_list.pop()+skip}).values()
            if i > 0:
                X, L, skip = self.UpScales[i-1](X,L,skip)

        
        L = F.relu(skip)
        L = F.relu(self.LastConvHidden(L))
        L = self.LastConv(L)
        return L

