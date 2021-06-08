cd('/zhome/75/a/138421/Desktop/BachelorProject/Data_Generation')
texture_folder = 'Original_Brodatz/';

dir_list = dir([texture_folder,'D*.gif']);
textures = zeros(640,640,length(dir_list));
for i=1:length(dir_list)
    textures(:,:,i) = im2double(imread([texture_folder,dir_list(i).name]));
end
size(textures)
%%

opts = get_random_opts_struct('hard8',4);
opts.dims = [64,128];
opts.t.big_noise_std = 15/4;
opts.t.big_noise_qts = [0.95,0.98];
opts.t.big_noise_bool = 1;
opts.t.big_noise_texture = 1;
[sample,s_walk,walk,layer_pic,label] = generate_sample(textures,opts);
subplot(2,2,1)
plot(walk,'linewidth',2)
title('Random walks')
subplot(2,2,3)
plot(s_walk,'linewidth',2)
title('Stacked random walks')
subplot(1,2,2)
imshow(sample), hold on
plot(s_walk(1:end/2,:))
hold on
%%
k = 20;
sample = cell(k,1);
for i=1:k
    opts = get_random_opts_struct('hard9',randi(6));
    %opts.t.intensity_interval = 1;
    %opts.t.std_mult = 0.01;
    opts.t.big_noise_bool = 1;
    opts.t.big_noise_texture = (rand()>0.2)*(rand())^(rand()>0.5);
    sample{i} = generate_sample(textures,opts);
end
figure;
montage(sample,'thumbnailsize',opts.dims,'size',[4,5])
%%
name = 'hard9';
num_samples = [6000,200,200];
T = length(num_samples);
layer_vec = 1:5;
folders = ["/train/","/test/","/validation/"];
opts_big = cell(T,1);
for t=1:T
    opts_big{t} = repmat(get_random_opts_struct(name,1),length(layer_vec),num_samples(t));
end
k = 0;
for t=1:T
for n=1:length(layer_vec)
n_layers = layer_vec(n);
for i=1:num_samples(t)
    opts_big{t}(n,i) = get_random_opts_struct(name,n_layers);
    flag = 1;
    for tries=1:3
        [sample,~,~,~,label] = generate_sample(textures,opts_big{t}(n,i));
        if abs(max(label(:))-n_layers)<1e-10, break;end
        warning("wrong layer number, retrying. try#"+tries)
    end
    imwrite(sample,"data/"+name+folders(t)+"/pic_l"+n_layers+"_"+i+".png")
    imwrite(uint8(round(label*16)),"data/"+name+folders(t)+"/label_l"+n_layers+"_"+i+".png")
    k = k + 1;
    if mod(k,500)==0
        disp(k)
    end
end
end
opts = opts_big{t};
save("data/"+name+folders(t)+"/opts.mat",'opts')
end
