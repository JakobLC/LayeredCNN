function [layer_pic,label,s] = get_layer_pic(s,opts)
if opts.l.vertical_mod < 0
    s = s/max(s,[],'all')*(opts.dims(1)-1+opts.l.vertical_mod)+randi(-opts.l.vertical_mod+1);
    imsize = opts.dims;
elseif opts.l.vertical_mod > 0
    s = s/max(s,[],'all')*(opts.dims(1)-1)+1;
    imsize = opts.dims+[opts.l.vertical_mod,0];
else
    s = s/max(s,[],'all')*(opts.dims(1)-1)+1;
    imsize = opts.dims;
end


layer_pic = zeros(imsize);
for i=1:size(s,2)
    s_floor = floor(s(:,i));
    s_ceil = ceil(s(:,i));
    layer_pic(sub2ind(imsize,s_floor(:),(1:imsize(2))')) = ...
    layer_pic(sub2ind(imsize,s_floor(:),(1:imsize(2))')) + s_ceil-s(:,i);
    layer_pic(sub2ind(imsize,s_ceil(:),(1:imsize(2))')) = ...
    layer_pic(sub2ind(imsize,s_ceil(:),(1:imsize(2))')) + s(:,i)-s_floor+(s_floor==s_ceil);
end
label = cumsum(layer_pic);
if opts.l.max_local_mix ~= 0
    g = get_gauss_filt(opts.l.local_layer_std);
    layer_pic_local = imfilter(label,g','replicate')+0.5;
else
    layer_pic_local = zeros(imsize);
end
if opts.l.min_local_mix ~= 1
    layer_pic_tmp = zeros(imsize);
    for i=1:imsize(2)
        layer_pic_tmp(:,i) = interp1([0.9999,s(i,:)+0.000001*((1:size(s,2))-round(size(s,2)/2)),imsize(1)+0.0001],0:size(s,2)+1,1:imsize(1))';
    end
%    layer_pic = round(label);
%    layer_pic_tmp = zeros(size(layer_pic));
%    for i=0:size(s,2)
%        mask = layer_pic==i;
%        sum_mask = sum(mask);
%        error_adjust = (layer_pic>i).*(sum_mask==0);
%        sum_mask(sum_mask==0) = 1;
%        layer_pic_tmp = layer_pic_tmp + cumsum(mask)./sum_mask + error_adjust;
%    end
    layer_pic_global = layer_pic_tmp;
else
    layer_pic_global = zeros(imsize);
end
t = linspace(0,1,opts.dims(2));
width_mix = interp1([0;sort(rand(opts.l.num_mix,1));1],rand(opts.l.num_mix+2,1),t);
width_mix = opts.l.min_local_mix + width_mix*(opts.l.max_local_mix-opts.l.min_local_mix);
%g = get_gauss_filt(2);
d = (length(g)-1)/2;
layer_pic = width_mix.*layer_pic_local+(1-width_mix).*(layer_pic_global);
if opts.l.vertical_mod > 0
    r_int = randi(opts.l.vertical_mod+1)-1;
    layer_pic = layer_pic((1:opts.dims(1))+r_int,:);
    label = label((1:opts.dims(1))+r_int,:);
end
end