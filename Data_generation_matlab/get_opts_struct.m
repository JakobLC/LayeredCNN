function opts = get_opts_struct()
opts = struct;
opts.num_walks = 5; %number of layer seperations
opts.dims = [512,512]; %image dimensions

%layer options
opts.l.walk_var = 0.2; %variability of vertical height between random walks,[0,1]
opts.l.dist = 1; %base distance between walks
opts.l.num_jumps = 20; %number of vertical jumps for general curve trends
opts.l.num_noise = 200; %number of points used for noise
opts.l.noise_ratio = 0.3; %ratio of noise [0,1] where the rest is the walk curve
opts.l.smoothing_ratio = 1/100; %std of the smoothing gaussian given as image width ratio
opts.l.seq_stack_p = 0.8; %percentage of seq walk stacking being new [0,1]
opts.l.structure_method = randi(3); %method for structuring the walks, 'trans', 'stack', 'swap'
opts.l.border_rate = 0.5; %rate of thresholds that contain a new layer at the border
opts.l.min_local_mix = 0.1; %
opts.l.max_local_mix = 0.8; %
opts.l.num_mix = 3; %
opts.l.local_layer_std = 3; %local std width of gaussian filter in pixels used for local border
opts.l.layer_width = 0.1; %width of the gradient between layers
opts.l.layer_width_border = 0.1; %width of the gradient between thin layers
opts.l.vertical_mod = 0; %integer of either additional space or reduced vertical pixels height layers will appear in

%texture options
opts.t.rotate_bool = 1; %should textures be rotated
opts.t.max_zoom = 4; %maximum zoom multiplier for textures, 1=no zoom
opts.t.same_rate = 0.3; %rate of same textures used for different layers [0,1]
opts.t.inversion_rate = 0.2; %rate of intensity inversion for textures
opts.t.derivative_rate = 0; %rate of texture intensities filtered with derivative gaussian
opts.t.min_std = 0.2; %min multiplication of std of the textures possible [0,1]
opts.t.std_mult = 1; %multiplier for the std of the textures
opts.t.intensity_interval = 0.8; %size of the interval of mean texture intensity used
opts.t.pepper_noise = 0.2; %uniform interval length of random pepper noies 
opts.t.pepper_rate = 0.1; %rate of pixels affected by pepper noise
opts.t.noise_size_mult = 0.02; %minimum size multiplier (in terms of image size) for noise ]0,1]. smaller = big noise blobs
opts.t.gauss_noise = 0.05; %std for added gaussian noise
opts.t.layer_dependent_noise = 0.5; %ratio in [0,1] of how layer-dependent the noise should be 0=no layer dependence
opts.t.num_layer_dependent = 10; %number of sampling points used for layer-dependent noise
opts.t.blur_std = 3; %applies random texture blur with std in range [0, blur_std] with [0,0.4]=no blur

end