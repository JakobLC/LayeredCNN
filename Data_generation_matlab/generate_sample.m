function [sample,s_walk,walk,layer_pic,label] = generate_sample(textures,opts)

walk = random_walk(opts);

[s_walk,walk] = structure_walks(walk,opts);

[layer_pic,label,s_walk] = get_layer_pic(s_walk,opts);

sample = combine_textures_with_layers(textures,layer_pic,opts);
end

function [s_walk,walk] = structure_walks(walk,opts)
d = opts.l.dist;
random_vec = (rand(1,opts.num_walks)*2-1)*opts.l.walk_var+1;
switch opts.l.structure_method
    case 'trans'
        s_walk = walk.*random_vec;
        s_walk(:,2:end) = s_walk(:,2:end)-cumsum(min(s_walk(:,2:end)-d-s_walk(:,1:end-1),[],1));
    case 'stack'
        p = opts.l.seq_stack_p;
        order = {1:ceil(opts.num_walks/2)-1,...
                ceil(opts.num_walks/2),...
                ceil(opts.num_walks/2)+(1:floor(opts.num_walks/2))};
        order{1} = order{1}(end:-1:1); 
        s_walk(:,order{2}) = random_vec(order{2})*walk(:,order{2});
        for i=order{1}
            s_walk(:,i) = s_walk(:,i+1) - random_vec(i)*(p*walk(:,i)+(1-p)*mean(walk(:,i)));
        end
        for i=order{3}
            s_walk(:,i) = s_walk(:,i-1) + random_vec(i)*(p*walk(:,i)+(1-p)*mean(walk(:,i)));
        end
        s_walk = s_walk - min(s_walk,[],'all');
        if opts.l.collapse_rate>0
            d = d*(rand()<0.5);
        end
        s_walk = s_walk + linspace(0,d*(opts.num_walks-1),opts.num_walks);
    case 'swap'
        s_walk = walk.*random_vec;
        s_walk(:,2:end) = s_walk(:,2:end)+3*d.*(1:opts.num_walks-1);
        s_walk = sort(s_walk,2);
        s_walk = s_walk + linspace(0,d*(opts.num_walks-1),opts.num_walks);
        g = get_gauss_filt(opts.dims(2)*opts.l.smoothing_ratio)';
        s_walk = imfilter(s_walk,g,'replicate');
        border_dist = (size(walk,1)-opts.dims(2))/2;
        s_walk = s_walk(border_dist+1:end-border_dist,:);
        
        walk = imfilter(walk,g,'replicate');
        walk = walk(border_dist+1:end-border_dist,:);
end
end

function walk = random_walk(opts)

walk_dims = [2*round(opts.dims(2)*(opts.l.num_jumps+2)/opts.l.num_jumps*0.5),opts.num_walks];
t = linspace(0,1,walk_dims(1));
R_jumps = zeros(walk_dims);
R_noise = zeros(walk_dims);
for i=1:opts.num_walks
    R_jumps(:,i) = (1-opts.l.noise_ratio)*...
        interp1([0;sort(rand(opts.l.num_jumps,1));1],cumsum(randn(opts.l.num_jumps+2,1)),t);
    R_noise(:,i) = opts.l.noise_ratio*...
        interp1([0;sort(rand(opts.l.num_noise,1));1],randn(opts.l.num_noise+2,1),t);
end

walk = R_jumps+R_noise;
if ~strcmpi('swap',opts.l.structure_method)
    g = get_gauss_filt(opts.dims(2)*opts.l.smoothing_ratio)';
    walk = imfilter(walk,g);
    border_dist = (walk_dims(1)-opts.dims(2))/2;
    walk = walk(border_dist+1:end-border_dist,:);
end
walk = walk-min(walk,[],1);
if opts.l.collapse_rate > 0 && strcmpi('stack',opts.l.structure_method)
    for i=[1:ceil(opts.num_walks/2)-1,ceil(opts.num_walks/2)+1:opts.num_walks]
        if rand() < opts.l.collapse_rate
            [~,min_idx] = min(walk(:,i));
            col_add = round(opts.l.collapse_length*size(walk,1));
            walk_tmp = [walk(1:min_idx,i);zeros(col_add,1);walk(min_idx+1:end,i)];
            walk_tmp = walk_tmp(ceil(col_add/2)+1:end-floor(col_add/2));
            walk(:,i) = walk_tmp;
        end
    end
end
end

function sample = combine_textures_with_layers(textures,layer_pic,opts)
[new_textures,threshold_bool] = modify_textures(textures,opts);
layer_pic_3d = zeros([opts.dims,opts.num_walks+1+sum(threshold_bool)]);
%sigmoid = @(x,L,w) 1./(1+exp(-(x-L)/w));
gaussian = @(x,L,w) exp(-0.5*((x-L)/w).^2);
w = opts.l.layer_width;
for i=1:(opts.num_walks+1)
    layer_pic_3d(:,:,i) = min(sigmoid(layer_pic,i-1-10*(i==1),w),...
                              1-sigmoid(layer_pic,i+10*(i==(opts.num_walks+1)),w));
end
threshold_idx = find(threshold_bool);
w = opts.l.layer_width_border;
for i=opts.num_walks+2:opts.num_walks+1+sum(threshold_bool)
    layer_pic_3d(:,:,i) = gaussian(layer_pic,threshold_idx(i-opts.num_walks-1),w);
end
layer_pic_3d(:,:,1:(opts.num_walks+1)) = layer_pic_3d(:,:,1:(opts.num_walks+1))...
                                .*(1-sum(layer_pic_3d(:,:,(opts.num_walks+2):end),3));
sample = sum(new_textures(:,:,1:opts.num_walks+1+sum(threshold_bool)).*layer_pic_3d,3);
if opts.t.big_noise_bool
    if opts.t.big_noise_texture>0
        big_noise_t = new_textures(:,:,opts.num_walks+3+sum(threshold_bool));
        qts = quantile(big_noise_t,[0.1,0.9],'all');
        big_noise_t = (big_noise_t-qts(1))/(eps+qts(2)-qts(1));
        big_noise_t(big_noise_t>1) = 1;
        big_noise_t(big_noise_t<0) = 0;
        big_noise_t = big_noise_t*opts.t.big_noise_texture+(1-opts.t.big_noise_texture);
    else
        big_noise_t = 1;
    end
    g = get_gauss_filt(opts.t.big_noise_std);
    d = (length(g)-1)/2;
    alpha_big_noise = randn(opts.dims+2*d);
    alpha_big_noise = imfilter(imfilter(alpha_big_noise,g),g');
    alpha_big_noise = alpha_big_noise(1+d:end-d,1+d:end-d);
    qts = quantile(alpha_big_noise,opts.t.big_noise_qts,'all');
    alpha_big_noise = (alpha_big_noise-qts(1))/(eps+qts(2)-qts(1));
    alpha_big_noise(alpha_big_noise<0) = 0;
    alpha_big_noise(alpha_big_noise>1) = 1;
    alpha_big_noise = alpha_big_noise.*big_noise_t;
    sample = sample.*(1-alpha_big_noise)+alpha_big_noise.*new_textures(:,:,opts.num_walks+2+sum(threshold_bool));
end

if opts.t.layer_dependent_noise > 0
    trans = (1-opts.t.layer_dependent_noise)+...
    opts.t.layer_dependent_noise*interp1(sort([0;rand(opts.t.num_layer_dependent,1);1]),...
        rand(opts.t.num_layer_dependent+2,1),layer_pic/(opts.num_walks+1));
else
    trans = ones(size(layer_pic));
end
noise_size = round(size(sample)*opts.t.noise_size_mult);
noise_gauss = randn(noise_size);
noise_pepper = (rand(noise_size)-0.5)...
    .*(rand(noise_size)<(opts.t.pepper_rate*rand()));
if any(size(sample)~=noise_size)
    noise_gauss = imresize(noise_gauss,size(sample));
    noise_pepper = imresize(noise_pepper,size(sample));
end

sample = sample + opts.t.gauss_noise.*trans.*noise_gauss;
sample = sample + opts.t.pepper_noise.*trans.*noise_pepper;
sample(sample>1) = 1;
sample(sample<0) = 0;
end

function [new_textures,threshold_bool] = modify_textures(textures,opts)
threshold_bool = rand(opts.num_walks,1)<=opts.l.border_rate;
N = opts.num_walks+1+sum(threshold_bool)+opts.t.big_noise_bool*(1+(opts.t.big_noise_texture>0));
same_num = ceil(opts.t.same_rate*N);

same_textures = repmat(textures(:,:,randi(size(textures,3))),1,1,same_num);

different_textures = textures(:,:,randperm(size(textures,3),N-same_num));

textures = cat(3,same_textures,different_textures);
textures = textures(:,:,randperm(size(textures,3)));

width = size(textures,1);
b_r = ceil(((1-sqrt(2)/2)*width)/2);
new_width = width-b_r*2;

new_textures = zeros([opts.dims,size(textures,3)]);
for i=1:size(textures,3)
    tmp = textures(:,:,i);
    if opts.t.inversion_rate>rand()
        tmp = 2*mean(tmp,'all')-tmp;
    end
    if opts.t.derivative_rate>rand()
        tmp = deriv_modification(tmp,opts);
    end
    if opts.t.blur_std>0
        blur_std = rand()*opts.t.blur_std;
        if blur_std<0.4
            g = 1;
        else
            g = get_gauss_filt(blur_std);
        end
        tmp = imfilter(imfilter(tmp,g,'replicate'),g','replicate');
    end
    if opts.t.rotate_bool
        tmp = imrotate(tmp,randi(360),'bicubic','crop');
    end
    tmp = tmp(b_r+1:end-b_r,b_r+1:end-b_r);
    new_shape = ceil(opts.dims/max(opts.dims)*new_width*...
        (rand()*(1-1/opts.t.max_zoom)+1/opts.t.max_zoom));
    tmp = tmp((1:new_shape(1))+randi(new_width-new_shape(1)+1)-1,...
              (1:new_shape(2))+randi(new_width-new_shape(2)+1)-1);
    new_textures(:,:,i) = imresize(tmp,opts.dims,'bicubic');
end
old_means = mean(new_textures,[1,2]);
old_stds = std(new_textures,1,[1,2]);
new_means = old_means*(1-opts.t.intensity_interval)+rand(size(old_means))*opts.t.intensity_interval;
new_stds = old_stds.*(opts.t.min_std+rand(size(old_stds))*(1-opts.t.min_std));
edge_modifier = @(x,k,d) d+(1-d)*min(min(k*x,k-k*x),1);
new_stds_edge_modifier = edge_modifier(new_means,3,0.2).*new_stds;
new_textures = (new_textures-old_means)./old_stds.*new_stds_edge_modifier*opts.t.std_mult+new_means;
end

function d_pic = deriv_modification(pic,opts)
r = rand();
g = get_gauss_filt(opts.dims(2)*opts.l.smoothing_ratio,1)';
if r < 0.25 %abs
    d_pic = sqrt(imfilter(pic,g,'replicate').^2+imfilter(pic,g','replicate').^2);
elseif r < 0.5 % X
    d_pic = imfilter(pic,g,'replicate');
elseif r < 0.75 % Y
    d_pic = imfilter(pic,g','replicate');
else %signed abs
    A = imfilter(pic,g,'replicate');
    B = imfilter(pic,g','replicate');
    d_pic = sqrt(A.^2+B.^2);
    d_pic = d_pic.*sign(A.*B);
end
d_pic = (d_pic-mean(d_pic,'all'))/std(d_pic,1,'all');
if rand()<0.5
    d_pic = -d_pic;
end
d_pic = mean(pic,'all')+d_pic*std(pic,1,'all');
end

function y = sigmoid(x,L,w)
y = 1./(1+exp(-(x-L)/w));
sigmoid_ratio = min(max(5-abs(x-L)*10,0),1);
y = y.*sigmoid_ratio+(1-sigmoid_ratio).*(x>L);
end

function y = fake_gaussian(x,L,w)
y = ((x-L)*w).^4+1;
y = y.^(-1-exp(y-2));
y(y<0.001) = 0;
end