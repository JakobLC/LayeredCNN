import glob
import os
import torchvision.transforms as transforms
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
import PIL.Image as Image

class Layers_Dataset(torch.utils.data.Dataset):  
    def __init__(self, opts, num_layers,train_bool=True, _transform=transforms.ToTensor(),
                 data_path='drive/My Drive/Bachelor_proj/data_easy/train',resize=1):

        self.opts = opts 
        self._transform = _transform
        if num_layers > 0:
            self.im_paths = glob.glob(os.path.join(data_path, "pic_l"+str(num_layers)+"_*.png"))
            self.gt_paths = glob.glob(os.path.join(data_path, "label_l"+str(num_layers)+"_*.png"))
        else:
            self.im_paths = glob.glob(os.path.join(data_path, "pic_*.png"))
            self.gt_paths = glob.glob(os.path.join(data_path, "label_*.png"))
        self.train_bool = train_bool

        self.im_paths.sort()
        self.gt_paths.sort()
        self.resize = resize
        
    def __len__(self):
        return len(self.im_paths)
    
    def augment_data(self, im, gt):
        if self.resize is not None:
            if np.size(self.resize)==1:
                d2,d1 = im.size
                resize_func = transforms.Resize(size=(int(d1//self.resize),int(d2//self.resize)))
            else:
                resize_func = transforms.Resize(size=self.resize)
            im = resize_func(im)
            gt = resize_func(gt)


        
        im = self._transform(im)
        gt = self._transform(gt)   

        gt = gt*255*0.0625
        k = int(gt.max().round())
        size_vec = torch.tensor(gt.shape)
        size_vec[0] = k
        gt_big = torch.zeros(torch.Size(size_vec))
        for i in range(k):
            tmp = gt-i
            tmp[tmp<0] = 0
            tmp[tmp>1] = 1
            gt_big[i,:,:] = tmp
        gt = gt_big
        
        if self.train_bool:
            if np.random.rand() > 0.5:
                im = torch.flip(im,[2])
                gt = torch.flip(gt,[2])

            if np.random.rand() > 0.5:
                im = torch.flip(im,[0,1])
                gt = torch.flip(gt,[0,1])
                gt = 1-gt
        return im, gt
    
    def warp_border(self, label):
        xt = np.arange(0,label.shape[2])
        
        for i in range(label.shape[1]):
            if self.opts["warp_border"]["bool"]:
                tmp = label[0,i,:,:].numpy()
                

                horz_std = self.opts["warp_border"]["h_std"][0]\
+np.random.rand(1)*(self.opts["warp_border"]["h_std"][1]-
                    self.opts["warp_border"]["h_std"][0])
                vert_std = self.opts["warp_border"]["v_std"][0]\
+np.random.rand(1)*(self.opts["warp_border"]["v_std"][1]-
                    self.opts["warp_border"]["v_std"][0])

                t = gaussian_filter(np.random.randn(tmp.shape[1]),
                                    sigma=horz_std,mode='nearest')
                t *= vert_std/t.std()
                
                tmp = np.array([np.interp(xt+t[j],
                            xt,tmp[:,j]) for j in range(tmp.shape[1])]).T
                
            label[0,i,:,:] = torch.from_numpy(tmp)
        return label
        
    
    def __getitem__(self, idx):
        im_path = self.im_paths[idx]
        gt_path = self.gt_paths[idx]

        im = Image.open(im_path)
        gt = Image.open(gt_path)
        
        im, gt = self.augment_data(im,gt)

        gt.unsqueeze_(0)
        label = gt.clone()
        
        if self.opts["warp_border"]["bool"]:
            label = self.warp_border(label)
        
        return im, label, gt