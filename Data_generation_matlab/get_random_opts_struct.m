function opts = get_random_opts_struct(mode,n_layers)
if nargin == 0
    mode = 'hard';
    n_layers = randi(6);
elseif nargin == 1 && strcmpi(mode,'hard')
    n_layers = randi(6);
end
switch mode
    case 'hard'
        opts = struct;
        opts.num_walks = n_layers; %number of layer seperations
        opts.dims = [256,512]; %image dimensions

        %layer options
        opts.l.walk_var = 0.1+rand()*0.9; %variability of vertical height between random walks,[0,1]
        opts.l.dist = rand()*(rand()<0.25); %base distance between walks
        opts.l.num_jumps = 2+randi(50); %number of vertical jumps for general curve trends
        opts.l.num_noise = 10+randi(256); %number of points used for noise
        opts.l.noise_ratio = rand()*rand()*0.6; %ratio of noise [0,1] where the rest is the walk curve
        opts.l.smoothing_ratio = 1/(10^(1.5+rand())); %std of the smoothing gaussian given as image width ratio
        opts.l.seq_stack_p = 0.2+rand()*0.8; %percentage of seq walk stacking being new [0,1]
        opts.l.structure_method = randi(3); %method for structuring the walks, 'trans', 'stack', 'swap'
        opts.l.border_rate = rand()^3; %rate of thresholds that contain a new layer at the border
        opts.l.min_local_mix = rand(); %minimum ratio of local layer pic, [0,1]
        opts.l.max_local_mix = opts.l.min_local_mix+(1-opts.l.min_local_mix)*rand(); %maximum ratio of local layer pic, [0,1]
        opts.l.num_mix = 2+randi(5); %number of sampled points used in width-dependent local/global layer pic mix
        opts.l.local_layer_std = 3*(0.5+1.5*rand()); %local std width of gaussian filter in pixels used for local border
        opts.l.layer_width = 0.1*(0.5+1.5*rand()); %width of the gradient between layers
        opts.l.layer_width_border = 0.1*(0.5+1.5*rand()); %width of the gradient between thin layers
        opts.l.vertical_mod = (-randi(16*7)+12*(opts.num_walks-1))*(rand()<0.9); %integer of either additional space or reduced vertical pixels height layers will appear in
        opts.l.collapse_rate = (rand()<0.4)*rand(); %when 'stack' structuring method is used, chance of layer collapse
        opts.l.collapse_length = 0.05+rand()*0.5; %length of layer collapse
        
        %texture options
        opts.t.rotate_bool = rand()<0.9; %should textures be rotated
        opts.t.max_zoom = 1+rand()*5; %maximum zoom multiplier for textures, 1=no zoom
        opts.t.same_rate = rand()*(rand()<0.1); %rate of same textures used for different layers [0,1]
        opts.t.inversion_rate = 0.2; %rate of intensity inversion for textures
        opts.t.derivative_rate = 0; %rate of texture intensities filtered with derivative gaussian
        opts.t.min_std = 0.2+rand()*0.5; %min multiplication of std of the textures possible [0,1]
        opts.t.std_mult = 0.2+rand(); %multiplier for the std of the textures
        opts.t.intensity_interval = 0.1+rand()*0.85; %size of the interval of mean texture intensity used
        opts.t.pepper_noise = (rand()<0.3)*(0.1+rand()*0.3); %uniform interval length of random pepper noies 
        opts.t.pepper_rate = rand()*0.5; %rate of pixels affected by pepper noise
        opts.t.noise_size_mult = 0.3+0.7*rand(); %minimum size multiplier (in terms of image size) for noise ]0,1]. smaller = big noise blobs
        opts.t.gauss_noise = 0.1*rand()*(rand()<0.4); %std for added gaussian noise
        opts.t.layer_dependent_noise = rand(); %ratio in [0,1] of how layer-dependent the noise should be 0=no layer dependence
        opts.t.num_layer_dependent = 2+randi(10); %number of sampling points used for layer-dependent noise
        opts.t.blur_std = rand()*3; %applies random texture blur with std in range [0, blur_std] with [0,0.4]=no blur
    case 'easy'
        opts = struct;
        opts.num_walks = 4; %number of layer seperations
        opts.dims = [256,256]; %image dimensions

        %layer options
        opts.l.walk_var = 0.5; %variability of vertical height between random walks,[0,1]
        opts.l.dist = 2; %base distance between walks
        opts.l.num_jumps = 15; %number of vertical jumps for general curve trends
        opts.l.num_noise = 100; %number of points used for noise
        opts.l.noise_ratio = 0.2; %ratio of noise [0,1] where the rest is the walk curve
        opts.l.smoothing_ratio = 1/100; %std of the smoothing gaussian given as image width ratio
        opts.l.seq_stack_p = 1; %percentage of seq walk stacking being new [0,1]
        opts.l.structure_method = randi(3); %method for structuring the walks, 'trans', 'stack', 'swap'
        opts.l.border_rate = 0.3; %rate of thresholds that contain a new layer at the border
        opts.l.min_local_mix = 0.2; %minimum ratio of local layer pic, [0,1]
        opts.l.max_local_mix = 0.8; %maximum ratio of local layer pic, [0,1]
        opts.l.num_mix = 5; %number of sampled points used in width-dependent local/global layer pic mix
        opts.l.local_layer_std = 3; %local std width of gaussian filter in pixels used for local border
        opts.l.layer_width = 0.1; %width of the gradient between layers
        opts.l.layer_width_border = 0.1; %width of the gradient between thin layers
        opts.l.vertical_mod = -20; %integer of either additional space or reduced vertical pixels height layers will appear in
        opts.l.collapse_rate = 1; %when 'stack' structuring method is used, chance of layer collapse
        opts.l.collapse_length = 0.5; %length of layer collapse

        %texture options
        opts.t.rotate_bool = rand()<0.9; %should textures be rotated
        opts.t.max_zoom = 3; %maximum zoom multiplier for textures, 1=no zoom
        opts.t.same_rate = 0; %rate of same textures used for different layers [0,1]
        opts.t.inversion_rate = 0; %rate of intensity inversion for textures
        opts.t.derivative_rate = 0; %rate of texture intensities filtered with derivative gaussian
        opts.t.min_std = 0.5; %min multiplication of std of the textures possible [0,1]
        opts.t.std_mult = 0.2; %multiplier for the std of the textures
        opts.t.intensity_interval = 0.8; %size of the interval of mean texture intensity used
        opts.t.pepper_noise = 0; %uniform interval length of random pepper noies 
        opts.t.pepper_rate = 0; %rate of pixels affected by pepper noise
        opts.t.noise_size_mult = 1; %minimum size multiplier (in terms of image size) for noise ]0,1]. smaller = big noise blobs
        opts.t.gauss_noise = 0; %std for added gaussian noise
        opts.t.layer_dependent_noise = 0; %ratio in [0,1] of how layer-dependent the noise should be 0=no layer dependence
        opts.t.num_layer_dependent = 5; %number of sampling points used for layer-dependent noise
        opts.t.blur_std = 0; %applies random texture blur with std in range [0, blur_std] with [0,0.4]=no blur
    case 'hard4'
        opts = struct;
        opts.num_walks = n_layers; %number of layer seperations
        opts.dims = [256,512]; %image dimensions

        %layer options
        opts.l.walk_var = 0.1+rand()*0.9; %variability of vertical height between random walks,[0,1]
        opts.l.dist = rand()*(rand()<0.25); %base distance between walks
        opts.l.num_jumps = 2+randi(50); %number of vertical jumps for general curve trends
        opts.l.num_noise = 10+randi(256); %number of points used for noise
        opts.l.noise_ratio = rand()*rand()*0.6; %ratio of noise [0,1] where the rest is the walk curve
        opts.l.smoothing_ratio = 0.01*(1.5+rand()-opts.l.num_jumps/50); %std of the smoothing gaussian given as image width ratio
        opts.l.seq_stack_p = 0.2+rand()*0.8; %percentage of seq walk stacking being new [0,1]
        opts.l.structure_method = randi(3); %method for structuring the walks, 'trans', 'stack', 'swap'
        opts.l.border_rate = rand()*rand()*(rand()<0.3); %rate of thresholds that contain a new layer at the border
        opts.l.min_local_mix = rand(); %minimum ratio of local layer pic, [0,1]
        opts.l.max_local_mix = opts.l.min_local_mix+(1-opts.l.min_local_mix)*rand(); %maximum ratio of local layer pic, [0,1]
        opts.l.num_mix = 2+randi(5); %number of sampled points used in width-dependent local/global layer pic mix
        opts.l.local_layer_std = 3*(0.5+1.5*rand()); %local std width of gaussian filter in pixels used for local border
        opts.l.layer_width = 0.1*(0.5+2*rand()); %width of the gradient between layers
        opts.l.layer_width_border = 0.1*(0.5+2*rand()); %width of the gradient between thin layers
        opts.l.vertical_mod = round(randn()*opts.dims(1)/10-(8-opts.num_walks)*opts.dims(1)/25.6); %integer of either additional space or reduced vertical pixels height layers will appear in
        opts.l.collapse_rate = (rand()<0.5)*rand()*rand(); %when 'stack' structuring method is used, chance of layer collapse
        opts.l.collapse_length = 0.05+rand()*rand()*0.8; %length of layer collapse
        
        %texture options
        opts.t.rotate_bool = rand()<0.9; %should textures be rotated
        opts.t.max_zoom = 1+rand()*5; %maximum zoom multiplier for textures, 1=no zoom
        opts.t.same_rate = rand()*(rand()<0.1); %rate of same textures used for different layers [0,1]
        opts.t.inversion_rate = 0.2; %rate of intensity inversion for textures
        opts.t.derivative_rate = 0; %rate of texture intensities filtered with derivative gaussian
        opts.t.min_std = 0.2+rand()*0.5; %min multiplication of std of the textures possible [0,1]
        opts.t.std_mult = 0.2+rand(); %multiplier for the std of the textures
        opts.t.intensity_interval = 0.1+rand()*0.85; %size of the interval of mean texture intensity used
        opts.t.pepper_noise = (rand()<0.3)*(0.1+rand()*rand()*0.4); %uniform interval length of random pepper noies 
        opts.t.pepper_rate = rand()*0.5; %rate of pixels affected by pepper noise
        opts.t.noise_size_mult = 0.3+0.7*rand(); %minimum size multiplier (in terms of image size) for noise ]0,1]. smaller = big noise blobs
        opts.t.gauss_noise = 0.15*rand()*rand()*(rand()<0.6); %std for added gaussian noise
        opts.t.layer_dependent_noise = rand(); %ratio in [0,1] of how layer-dependent the noise should be 0=no layer dependence
        opts.t.num_layer_dependent = 2+randi(10); %number of sampling points used for layer-dependent noise
        opts.t.blur_std = rand()*3; %applies random texture blur with std in range [0, blur_std] with [0,0.4]=no blur
    case 'hard6'
        opts = struct;
        opts.num_walks = n_layers; %number of layer seperations
        opts.dims = [256,512]; %image dimensions

        %layer options
        opts.l.walk_var = 0.1+rand()*0.85; %variability of vertical height between random walks,[0,1]
        opts.l.dist = exp(0.4*randn())*(rand()<0.95); %base distance between walks
        opts.l.num_jumps = 2+randi(48); %number of vertical jumps for general curve trends
        opts.l.num_noise = 9+randi(247); %number of points used for noise
        opts.l.noise_ratio = rand()*rand()*0.6; %ratio of noise [0,1] where the rest is the walk curve
        opts.l.smoothing_ratio = 0.01*(1.5+rand()-opts.l.num_jumps/50); %std of the smoothing gaussian given as image width ratio
        opts.l.seq_stack_p = 0.2+rand()*0.8; %percentage of seq walk stacking being new [0,1]
        opts.l.structure_method = sum([0,0.3,0.7]<rand()); %method for structuring the walks, 'trans', 'stack', 'swap'
        opts.l.border_rate = rand()*rand()*(rand()<0.3); %rate of thresholds that contain a new layer at the border
        opts.l.min_local_mix = rand();%rand(); %minimum ratio of local layer pic, [0,1]
        opts.l.max_local_mix = opts.l.min_local_mix+(1-opts.l.min_local_mix)*rand()*rand(); %maximum ratio of local layer pic, [0,1]
        opts.l.num_mix = 2+randi(5); %number of sampled points used in width-dependent local/global layer pic mix
        opts.l.local_layer_std = 3*(0.5+1.5*rand()); %local std width of gaussian filter in pixels used for local border
        opts.l.layer_width = 0.1*(0.5+1.5*rand()+0.5*rand()); %width of the gradient between layers
        opts.l.layer_width_border = 0.1*(0.5+2*rand()); %width of the gradient between thin layers
        opts.l.vertical_mod = round(randn()*opts.dims(1)/10-(8-opts.num_walks)*opts.dims(1)/25.6); %integer of either additional space or reduced vertical pixels height layers will appear in
        opts.l.collapse_rate = (rand()<0.5)*rand()*rand(); %when 'stack' structuring method is used, chance of layer collapse
        opts.l.collapse_length = 0.05+rand()*rand()*0.8; %length of layer collapse
        
        %texture options
        opts.t.rotate_bool = rand()<0.9; %should textures be rotated
        opts.t.max_zoom = 1+rand()*5; %maximum zoom multiplier for textures, 1=no zoom
        opts.t.same_rate = rand()*(rand()<0.1); %rate of same textures used for different layers [0,1]
        opts.t.inversion_rate = 0.2; %rate of intensity inversion for textures
        opts.t.derivative_rate = 0; %rate of texture intensities filtered with derivative gaussian
        opts.t.min_std = 0.2+rand()*0.5; %min multiplication of std of the textures possible [0,1]
        opts.t.std_mult = 0.2+rand(); %multiplier for the std of the textures
        opts.t.intensity_interval = 0.1+rand()*0.85; %size of the interval of mean texture intensity used
        opts.t.pepper_noise = (rand()<0.3)*(0.1+rand()*rand()*0.3); %uniform interval length of random pepper noies 
        opts.t.pepper_rate = rand()*0.5; %rate of pixels affected by pepper noise
        opts.t.noise_size_mult = 1;%0.3+0.7*rand(); %minimum size multiplier (in terms of image size) for noise ]0,1]. smaller = big noise blobs
        opts.t.gauss_noise = 0.1*rand()*rand()*(rand()<0.6); %std for added gaussian noise
        opts.t.layer_dependent_noise = rand()*rand(); %ratio in [0,1] of how layer-dependent the noise should be 0=no layer dependence
        opts.t.num_layer_dependent = 2+randi(10); %number of sampling points used for layer-dependent noise
        opts.t.blur_std = rand()*rand()*5; %applies random texture blur with std in range [0, blur_std] with [0,0.4]=no blur
    case 'hard7'
        opts = struct;
        opts.num_walks = n_layers; %number of layer seperations
        opts.dims = [256,512]; %image dimensions

        %layer options
        opts.l.walk_var = 0.1+rand()*0.85; %variability of vertical height between random walks,[0,1]
        opts.l.dist = exp(0.4*randn())*(rand()<0.95); %base distance between walks
        opts.l.num_jumps = 2+randi(48); %number of vertical jumps for general curve trends
        opts.l.num_noise = 9+randi(247); %number of points used for noise
        opts.l.noise_ratio = rand()*rand()*0.6; %ratio of noise [0,1] where the rest is the walk curve
        opts.l.smoothing_ratio = 0.01*(1.5+rand()-opts.l.num_jumps/50); %std of the smoothing gaussian given as image width ratio
        opts.l.seq_stack_p = 0.2+rand()*0.8; %percentage of seq walk stacking being new [0,1]
        opts.l.structure_method = sum([0,0.3,0.7]<rand()); %method for structuring the walks, 'trans', 'stack', 'swap'
        opts.l.border_rate = rand()*rand()*(rand()<0.3); %rate of thresholds that contain a new layer at the border
        opts.l.min_local_mix = rand();%rand(); %minimum ratio of local layer pic, [0,1]
        opts.l.max_local_mix = opts.l.min_local_mix+(1-opts.l.min_local_mix)*rand()*rand(); %maximum ratio of local layer pic, [0,1]
        opts.l.num_mix = 2+randi(5); %number of sampled points used in width-dependent local/global layer pic mix
        opts.l.local_layer_std = 3*(0.5+1.5*rand()); %local std width of gaussian filter in pixels used for local border
        opts.l.layer_width = 0.1*(0.5+1.5*rand()+0.5*rand()); %width of the gradient between layers
        opts.l.layer_width_border = 0.1*(0.5+2*rand()); %width of the gradient between thin layers
        opts.l.vertical_mod = round(randn()*opts.dims(1)/10-(8-opts.num_walks)*opts.dims(1)/25.6); %integer of either additional space or reduced vertical pixels height layers will appear in
        opts.l.collapse_rate = (rand()<0.5)*rand()*rand(); %when 'stack' structuring method is used, chance of layer collapse
        opts.l.collapse_length = 0.05+rand()*rand()*0.8; %length of layer collapse
        
        %texture options
        opts.t.rotate_bool = rand()<0.9; %should textures be rotated
        opts.t.max_zoom = 1+rand()*5; %maximum zoom multiplier for textures, 1=no zoom
        opts.t.same_rate = rand()*(rand()<0.1); %rate of same textures used for different layers [0,1]
        opts.t.inversion_rate = 0.2; %rate of intensity inversion for textures
        opts.t.derivative_rate = 0; %rate of texture intensities filtered with derivative gaussian
        opts.t.min_std = 0.2+rand()*0.5; %min multiplication of std of the textures possible [0,1]
        opts.t.std_mult = 0.1+rand()*rand()*1.2; %multiplier for the std of the textures
        opts.t.intensity_interval = 0.1+rand()*0.85; %size of the interval of mean texture intensity used
        opts.t.pepper_noise = (rand()<0.3)*(0.1+rand()*rand()*0.3); %uniform interval length of random pepper noies 
        opts.t.pepper_rate = rand()*0.5; %rate of pixels affected by pepper noise
        opts.t.noise_size_mult = 1;%0.3+0.7*rand(); %minimum size multiplier (in terms of image size) for noise ]0,1]. smaller = big noise blobs
        opts.t.gauss_noise = 0.1*rand()*rand()*(rand()<0.6); %std for added gaussian noise
        opts.t.layer_dependent_noise = rand()*rand(); %ratio in [0,1] of how layer-dependent the noise should be 0=no layer dependence
        opts.t.num_layer_dependent = 2+randi(10); %number of sampling points used for layer-dependent noise
        opts.t.blur_std = rand()*rand()*5; %applies random texture blur with std in range [0, blur_std] with [0,0.4]=no blur
    case 'hard8'
        opts = struct;
        opts.num_walks = n_layers; %number of layer seperations
        opts.dims = [256,512]; %image dimensions

        %layer options
        opts.l.walk_var = 0.1+rand()*0.85; %variability of vertical height between random walks,[0,1]
        opts.l.dist = exp(0.4*randn())*(rand()<0.95); %base distance between walks
        opts.l.num_jumps = 2+randi(48); %number of vertical jumps for general curve trends
        opts.l.num_noise = 9+randi(247); %number of points used for noise
        opts.l.noise_ratio = rand()*rand()*0.6; %ratio of noise [0,1] where the rest is the walk curve
        opts.l.smoothing_ratio = 0.01*(1.5+rand()*2-opts.l.num_jumps/50); %std of the smoothing gaussian given as image width ratio
        opts.l.seq_stack_p = 0.2+rand()*0.8; %percentage of seq walk stacking being new [0,1]
        opts.l.structure_method = sum([0,0.3,0.7]<rand()); %method for structuring the walks, 'trans', 'stack', 'swap'
        opts.l.border_rate = rand()*rand()*(rand()<0.3); %rate of thresholds that contain a new layer at the border
        opts.l.min_local_mix = rand();%rand(); %minimum ratio of local layer pic, [0,1]
        opts.l.max_local_mix = opts.l.min_local_mix+(1-opts.l.min_local_mix)*rand()*rand(); %maximum ratio of local layer pic, [0,1]
        opts.l.num_mix = 2+randi(5); %number of sampled points used in width-dependent local/global layer pic mix
        opts.l.local_layer_std = 3*(0.5+1.5*rand()); %local std width of gaussian filter in pixels used for local border
        opts.l.layer_width = 0.1*(0.5+1.5*rand()+0.5*rand()); %width of the gradient between layers
        opts.l.layer_width_border = 0.1*(0.5+2*rand()); %width of the gradient between thin layers
        opts.l.vertical_mod = round(randn()*opts.dims(1)/10-(8-opts.num_walks)*opts.dims(1)/25.6); %integer of either additional space or reduced vertical pixels height layers will appear in
        opts.l.collapse_rate = (rand()<0.5)*rand()*rand(); %when 'stack' structuring method is used, chance of layer collapse
        opts.l.collapse_length = 0.05+rand()*rand()*0.8; %length of layer collapse
        
        %texture options
        opts.t.rotate_bool = rand()<0.9; %should textures be rotated
        opts.t.max_zoom = 1+rand()*5; %maximum zoom multiplier for textures, 1=no zoom
        opts.t.same_rate = rand()*(rand()<0.1); %rate of same textures used for different layers [0,1]
        opts.t.inversion_rate = 0.2; %rate of intensity inversion for textures
        opts.t.derivative_rate = 0; %rate of texture intensities filtered with derivative gaussian
        opts.t.min_std = 0.2+rand()*0.5; %min multiplication of std of the textures possible [0,1]
        opts.t.std_mult = 0.1+rand()*rand()*1.2; %multiplier for the std of the textures
        opts.t.intensity_interval = 0.1+rand()*0.85; %size of the interval of mean texture intensity used
        opts.t.pepper_noise = (rand()<0.3)*(0.1+rand()*rand()*0.3); %uniform interval length of random pepper noies 
        opts.t.pepper_rate = rand()*0.5; %rate of pixels affected by pepper noise
        opts.t.noise_size_mult = 1;%0.3+0.7*rand(); %minimum size multiplier (in terms of image size) for noise ]0,1]. smaller = big noise blobs
        opts.t.gauss_noise = 0.1*rand()*rand()*(rand()<0.6); %std for added gaussian noise
        opts.t.layer_dependent_noise = rand()*rand(); %ratio in [0,1] of how layer-dependent the noise should be 0=no layer dependence
        opts.t.num_layer_dependent = 2+randi(10); %number of sampling points used for layer-dependent noise
        opts.t.blur_std = rand()*rand()*5; %applies random texture blur with std in range [0, blur_std] with [0,0.4]=no blur
        opts.t.big_noise_std = 3+rand()*7+rand()*20; %std for the alphamask for big noise
        opts.t.big_noise_qts = [0.9+(1-rand()*rand())*0.1,0]; %quantiles to be used as 0 and 1 for when rescaling the alphamask
        opts.t.big_noise_qts(2) = opts.t.big_noise_qts(1)+(1-opts.t.big_noise_qts(1))*(1-rand()*rand()); %second must be larger
        opts.t.big_noise_bool = rand()>0.2; %use big noise
        opts.t.big_noise_texture = (rand()>0.2)*(1-rand()^2); %use texture for big noise and how much? 0=none 1=max
        
    case 'hard9'
        opts = struct;
        #opts.num_walks = n_layers; %number of layer seperations
        opts.dims = [256,512]; %image dimensions

        %layer options
        #opts.l.walk_var = 0.1+rand()*0.85; %variability of vertical height between random walks,[0,1]
        #opts.l.dist = exp(0.4*randn())*(rand()<0.95); %base distance between walks
        #opts.l.num_jumps = 2+randi(48); %number of vertical jumps for general curve trends
        #opts.l.num_noise = 9+randi(247); %number of points used for noise
        #opts.l.noise_ratio = rand()*rand()*0.6; %ratio of noise [0,1] where the rest is the walk curve
        #opts.l.smoothing_ratio = 0.01*(1.5+rand()*2-opts.l.num_jumps/50); %std of the smoothing gaussian given as image width ratio
        #opts.l.seq_stack_p = 0.2+rand()*0.8; %percentage of seq walk stacking being new [0,1]
        #opts.l.structure_method = sum([0,0.3,0.7]<rand()); %method for structuring the walks, 'trans', 'stack', 'swap'
        #opts.l.border_rate = rand()*rand()*(rand()<0.3); %rate of thresholds that contain a new layer at the border
        opts.l.min_local_mix = rand();%rand(); %minimum ratio of local layer pic, [0,1]
        opts.l.max_local_mix = opts.l.min_local_mix+(1-opts.l.min_local_mix)*rand()*rand(); %maximum ratio of local layer pic, [0,1]
        opts.l.num_mix = 2+randi(5); %number of sampled points used in width-dependent local/global layer pic mix
        #opts.l.local_layer_std = 3*(0.5+1.5*rand()); %local std width of gaussian filter in pixels used for local border
        opts.l.layer_width = 0.1*(0.5+1.5*rand()+0.5*rand()); %width of the gradient between layers
        opts.l.layer_width_border = 0.1*(0.5+2*rand()); %width of the gradient between thin layers
        #opts.l.vertical_mod = round(randn()*opts.dims(1)/10-(8-opts.num_walks)*opts.dims(1)/25.6); %integer of either additional space or reduced vertical pixels height layers will appear in
        #opts.l.collapse_rate = (rand()<0.5)*rand()*rand(); %when 'stack' structuring method is used, chance of layer collapse
        #opts.l.collapse_length = 0.05+rand()*rand()*0.8; %length of layer collapse
        
        %texture options
        #opts.t.rotate_bool = rand()<0.9; %should textures be rotated
        #opts.t.max_zoom = 1+rand()*5; %maximum zoom multiplier for textures, 1=no zoom
        #opts.t.same_rate = rand()*(rand()<0.1); %rate of same textures used for different layers [0,1]
        #opts.t.inversion_rate = 0.2; %rate of intensity inversion for textures
        #opts.t.derivative_rate = 0; %rate of texture intensities filtered with derivative gaussian
        #opts.t.min_std = 0.2+rand()*0.5; %min multiplication of std of the textures possible [0,1]
        #opts.t.std_mult = 0.1+rand()*rand()*1.2; %multiplier for the std of the textures
        #opts.t.intensity_interval = 0.1+rand()*0.85; %size of the interval of mean texture intensity used
        #opts.t.pepper_noise = (rand()<0.3)*(0.1+rand()*rand()*0.3); %uniform interval length of random pepper noies 
        #opts.t.pepper_rate = rand()*0.5; %rate of pixels affected by pepper noise
        #opts.t.noise_size_mult = 1;%0.3+0.7*rand(); %minimum size multiplier (in terms of image size) for noise ]0,1]. smaller = big noise blobs
        #opts.t.gauss_noise = 0.1*rand()*rand()*(rand()<0.6); %std for added gaussian noise
        opts.t.layer_dependent_noise = rand()*rand(); %ratio in [0,1] of how layer-dependent the noise should be 0=no layer dependence
        opts.t.num_layer_dependent = 2+randi(10); %number of sampling points used for layer-dependent noise
        #opts.t.blur_std = rand()*rand()*5; %applies random texture blur with std in range [0, blur_std] with [0,0.4]=no blur
        #opts.t.big_noise_std = 3+rand()*7+rand()*20; %std for the alphamask for big noise
        #opts.t.big_noise_qts = [0.8+(1-rand()*rand())*0.2,0]; %quantiles to be used as 0 and 1 for when rescaling the alphamask
        #opts.t.big_noise_qts(2) = opts.t.big_noise_qts(1)+(1-opts.t.big_noise_qts(1))*(1-rand()*rand()); %second must be larger
        #opts.t.big_noise_bool = rand()<0.4; %use big noise
        #opts.t.big_noise_texture = (rand()>0.2)*(rand())^(rand()>0.5); %use texture for big noise and how much? 0=none 1=max
end
if isnumeric(opts.l.structure_method)
    methods = {'trans','stack','swap'};
    opts.l.structure_method = methods{opts.l.structure_method};
end
end