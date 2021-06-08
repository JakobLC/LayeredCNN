function g = get_gauss_filt(t,order_deriv,size)
% Function by Jakob L�nborg Christensen s183985 F2020
%
% Computes a 1d gaussian filter.
%
% Inputs:
%           t           = positive real numbers. The std of the filter
%           order_deriv = {0,1,2}. Number of times the derivative of the
%                         gaussian is taken. default: 0
%           size        = Naturals. Makes the returned vector N=2*size+1  
%                         long. default: ceil(4*t)
% Outputs:
%           g           = vector of size (1 x N). 1d gaussian filter kernel.
%
if nargin == 0
    order_deriv = 0;
    t = 1;
end
if nargin == 1
    order_deriv = 0;
end
if nargin < 3
    size = ceil(4*t);
end
x = -size:size;
g = exp(-x.^2/(2*t^2));
g = g/sum(g);
if order_deriv == 1
    g = g.*(-x/t^2);
elseif order_deriv == 2
    g = g.*(x.^2-t^2)/t^4;
end
end