#%% Get Dependencies
#you may have to install dependencies with e.g. "pip install torch"
import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from functions import sample_batch, OUtoSM, load_net, LCtoOU, automatic_init, plot_training_graph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available(): print("Warning: using CPU not GPU")
#%% Load a model (def++) 
#Write the folder location of the "example_script.py"
folder = "C:/Users/jakob/Desktop/DTU/Bachelor Project/saves/08_06_2021/"
best_net, net, opts = load_net("def_plus_plus",device,base_path=folder)
#%% Plot training graph of the loaded model
plot_training_graph(opts["losses"],log_bool=False,ylims=[(0.03,0.07),(0,3)])
#%% Generate some simple layer curves
image_dims = [64,128]

N_layers = 3
L_curve = torch.zeros((N_layers,image_dims[1]))
start_end = torch.linspace(image_dims[0]/(2*N_layers+3),
                        (1-1/(2*N_layers+3))*image_dims[0],
                        N_layers+1)

for i in range(N_layers):
    tmp = torch.rand(image_dims[1])
    tmp = torch.tensor(gaussian_filter(tmp,1+np.random.rand(1)*image_dims[1]/8))
    tmp = (tmp-tmp.min())/(tmp.max()-tmp.min())*(start_end[i+1]-start_end[i])+start_end[i]
                       
    L_curve[i,:] = (tmp+i)*image_dims[1]/(image_dims[1]+N_layers-1)
L_overunder = LCtoOU(L_curve,dims=image_dims)
L_onehot = OUtoSM(L_overunder)

plt.subplots(figsize=(10,10))

plt.plot(L_curve.T)
#%% Generate a synthetic image based on the layer curves
base_intensities = torch.rand(N_layers+1)
pixel_noise_std = 0.02 #How much Gaussian pixel noise should there be?
large_noise_std = 0.02 #How much larger area noise should there be?
large_noise_size = np.random.rand()*5+3 #how big should the larger area noise blobs be
blur_std = 1 #How sharp should the layers be

pixel_noise = torch.randn(image_dims)*pixel_noise_std
large_noise = torch.tensor(gaussian_filter(torch.randn(image_dims),sigma=large_noise_size))
large_noise = large_noise/(large_noise.std()+1e-14)*large_noise_std

I = torch.tensor(gaussian_filter((L_onehot*(base_intensities.reshape((1,1,-1,1,1)))).sum(2)
                    ,sigma=blur_std))+large_noise+pixel_noise
I = (I-I.min())/(I.max()-I.min())*2-1
plt.subplots(figsize=(10,10))
plt.imshow(I.squeeze(),cmap='gray')
plt.plot(L_curve.T)
#%% Sample the synthetic image, you can choose the initialization size
#How large should the initialization be?
initialization_size = 1 # X columns of pixels if an integer is used
#initialization_size = 0.2 # ratio of columns in [0,1] if a float is used

batch = (I,L_overunder,L_overunder)
L_hat, L_tmp, loss, L_curve_gt, L_curve_hat = sample_batch(net,device,
                                                           batch,opts,
                                                           label_pct=initialization_size)
L_curve_gt.squeeze_()
L_curve_hat.squeeze_()
#It should take up to 5 minutes for a 64x128 image on a cpu
#and down to a couple seconds for a fast GPU
#%% Plot the results!
fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(10,5))

axs[0].imshow(I.squeeze(),cmap='gray')
axs[0].plot(L_curve_gt.T)
axs[0].set_ylim([image_dims[0]-1,0])
axs[0].set_xlim([0,image_dims[1]-1])
axs[0].set_title("Image with ground truth curves")

axs[1].imshow(I.squeeze(),cmap='gray')
axs[1].plot(L_curve_hat.T)
axs[1].set_ylim([image_dims[0]-1,0])
axs[1].set_xlim([0,image_dims[1]-1])
axs[1].set_title("Image with predicted curves")
#%% Sample the synthetic with automatic layer initialization
selection = N_layers #if selection is int then it will select the best 


batch = (I.squeeze(0),[],[])
L_init, means, clusters, cost, clusters_cost, s_hat_init = automatic_init(net,
        device,batch,opts,selection=selection,n_r=10,cluster_dist=1.5)
batch = (I,L_init,L_init)
L_hat, L_tmp, loss, _, L_curve_hat = sample_batch(net,device,
                                                  batch,opts,
                                                  label_pct=initialization_size)
#%% Plot automatic initialization results
fig,axs = plt.subplots(nrows=1,ncols=3,figsize=(15,5))


axs[0].imshow(I.squeeze(),cmap='gray')
axs[0].plot(L_curve_gt.squeeze().T)
axs[0].set_ylim([image_dims[0]-1,0])
axs[0].set_xlim([0,image_dims[1]-1])
axs[0].set_title("Ground truth")

axs[1].imshow(I.squeeze(),cmap='gray')
axs[1].plot(s_hat_init.squeeze().T,color='r',alpha=0.5)
axs[1].scatter(2+np.zeros_like(means[clusters_cost<0.004]),means[clusters_cost<0.004],s=50,marker='<',color=[0,1,0])
axs[1].scatter(2+np.zeros_like(means[clusters_cost>=0.004]),means[clusters_cost>=0.004],s=50,marker='<',color=[0,0.4,1])
axs[1].set_yticks(means)
axs[1].set_yticklabels(clusters_cost.argsort().argsort()+1)
axs[1].set_ylim([image_dims[0]-1,0])
axs[1].set_xlim([0,image_dims[1]-1])
axs[1].set_title("Automatic initialization curves")

axs[2].imshow(I.squeeze(),cmap='gray')
axs[2].plot(L_curve_hat.squeeze().T)
axs[2].set_ylim([image_dims[0]-1,0])
axs[2].set_xlim([0,image_dims[1]-1])
axs[2].set_title("Prediction")
