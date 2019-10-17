function [K] = KernelMatrixHIK(X,Y,kernel,varargin)
% Calculate the matrix of kernel trick
% for each Kij = k(xi,yj), where xi = X(:,i), yj = Y(:,j);
% 
% Note: rewrite specially for HIK with the loop to free the RAM
% 
% Input:
% KernelMatrixHIK(X,Y,kernel,varargin)
%   X: input matrix, size of n*m, n bands m samples
%   Y: input matrix, size of n*m, n bands m samples
%   kernel: kernel type, such as
%       'poly': polynomal kernel, k(x,y) = (x.y)^d
%       'RBF':Gaussian kernel Radial Basic Function (RBF), k(x,y) = exp((-||x-y||^2)/2*cita^2)
%       'HIK':Histogram intersection kernel, k(x,y) = sum(min(x,y))
%   varagin: parameter of kernel
% 
% Output:
% K: the kernel matrix for input data X and Y
% 
% Note v1.1: calculate the kernel matrix with vectorization instead of
%   parellel loop
% 
% Note v1.2: add the kernel of HIK
% 
% Copyright (c) Chen Wu
% Version 1.3 2015/3/22

% size of matrix
[nx,mx] = size(X);
[ny,my] = size(Y);
if nx ~= ny
    error('the input matrics must have the same size');
end
n = nx;

K = zeros(mx,my);

% store parameter into one cell
cell_parameter = varargin{1};

% calculate the kernel matrix
switch kernel
    case 'linear'
        
    case 'poly'
        if numel(cell_parameter) < 1
            error('error parameter input of KernelTrick');
        end
        d = cell_parameter{1};
        K = (X'*Y).^d;
    case 'RBF'
        if numel(cell_parameter) < 1
            error('error parameter input of KernelTrick');
        end
        cita = cell_parameter{1};
        normXY = (ones(my,1)*sum(X.^2))'+(ones(mx,1)*sum(Y.^2))-2*X'*Y;
        K = exp(-normXY/(2*cita^2));
    case 'HIK'
        for i = 1:mx
            Xmatrix = X(:,i);
            Xmatrix = repmat(Xmatrix,1,my);
            K(i,:) = sum(min(Xmatrix,Y));
        end
    otherwise
        error('error kernel');         
end

end