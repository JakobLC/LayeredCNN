function label = generate_label_pic(layers,imsize)
n = length(layers);
label = zeros(imsize);
s = zeros(imsize(2),n);
for i=1:n
    s(:,i) = extend_segmentation_line(layers{i},imsize);
end

for i=1:n
    s_floor = floor(s(:,i));
    s_ceil = ceil(s(:,i));
    
    floor_bool = (s_floor < 1) | (imsize(1) < s_floor);
    ceil_bool = (s_ceil < 1) | (imsize(1) < s_ceil);
    
    s_floor = min(max(floor(s(:,i)),1),imsize(1));
    s_ceil = min(max(ceil(s(:,i)),1),imsize(1));
    
    label(sub2ind(imsize,s_floor(:),(1:imsize(2))')) = ...
    label(sub2ind(imsize,s_floor(:),(1:imsize(2))')) + (s_ceil-s(:,i)).*floor_bool;
    label(sub2ind(imsize,s_ceil(:),(1:imsize(2))')) = ...
    label(sub2ind(imsize,s_ceil(:),(1:imsize(2))')) + (s(:,i)-s_floor+(s_floor==s_ceil)).*floor_bool;
end
end