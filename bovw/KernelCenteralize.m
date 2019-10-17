function [Kc] = KernelCenteralize(K)
% Centeralize the kernel matrix
% 
% Input:
% KernelCenteralize(K)
%   K: kernel matrix
% 
% Output:
% Kc: centeralized kernel matrix
% 
% Copyright (c) Chen Wu
% Version 1.0 2014/11/25

% matrix size
[m,n] = size(K);

% centeralization
Kc = K-ones(m,m)/m*K-K*ones(n,n)/n+ones(m,m)/m*K*ones(n,n)/n;

end