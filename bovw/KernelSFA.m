function [SFeatures,V_cita,eigenv] = KernelSFA(DataX,DataY,TrainX,TrainY,kernel,varargin)
% Calculate kernel slow feature analysis, output the transformed features,
% tranformation matrix and eigenvalues
% the train dataset and the transformed dataset are different
% 
% Input:
% KernelSFA(DataX,DataY,TrainX,TrainY,kernel,varargin)
%   DataX: input data matrix X for transformation, size of n*md
%   DataY: input data matrix Y for transformation, size of n*md
%   TrainX: input train data matrix X for calculating the kernel
%       transformation matrix, size of n*mt
%   TrainY: input train data matrix X for calculating the kernel
%       transformation matrix, size of n*mt
%   kernel: kernel type, such as
%       'poly': polynomal kernel, k(x,y) = (x.y)^d
%       'gauss':Gaussian kernel, k(x,y) = exp((-||x-y||^2)/cita)
%       'RBF':Gaussian kernel Radial Basic Function (RBF), k(x,y) = exp((-||x-y||^2)/2*cita^2)
%       'HIK':Histogram intersection , k(x,y) = sum(min(x,y))
%   varagin: parameter of kernel
% 
% Note: DataX and DataY, TrainX and TrainY must be linked and have the same
%   size.
% 
% Output:
% [SFeatures,W,eigenv,Train_SFeatures]
%   SFeatures: the data matrix of slow features obtained by the
%       the difference of kernel transformation of DataX and DataY
%   W: the kernel transformation matrix
%   eigenv: eigenvalue from the SFA learning, less means slower and more
%       efficient
%   Train_SFeatures: the data matrix of slow features obtained by the
%       the difference of kernel transformation of TrainX and TrainY
% 
% Note: the output features, transformation matrix and eigenvalues have
%   all been sorted according to eigenvalues, less eigenvalue first
% 
% Note v1.1: change the order of matrix A and B, sort the eigenvalue
%   descend, use the function KernelMatrix v1.0
% 
% Note v1.2: when calculating kernel matrix, we don't use parallel loop.
% Instead, we use the vectorization calculation
% 
% Note v1.2.1: adaptive regularization parameter
% 
% Note v1.2.3: output SFeatures for Train data
% 
% Note v1.3: correct the kernel projection
% 
% Note v1.3.1: remove the projection of SFeatures for Train data
% 
% Copyright (c) Chen Wu
% Version 1.3.1 2015/4/24

% matrix size examination
if isequal(size(DataX),size(DataY)) == false
    error('input data X and Y must have the same size');
end
if isequal(size(TrainX),size(TrainY)) == false
    error('input train data X and Y must have the same size');
end
if isequal(size(DataX,1),size(TrainX,1)) == false
    error('input train data and tranformation data must have the same band size');
end

% size
[n,md] = size(DataX);
mt = size(TrainX,2);

% input data centeralization
% centeralize the input data with the statistical features of train data
%disp('data centeralization...');
meanTX = mean(TrainX,2);
meanTY = mean(TrainY,2);
stdTX = std(TrainX,0,2);
stdTY = std(TrainY,0,2);
stdTX(stdTX==0) = 1;
stdTY(stdTY==0) = 1;
TrainX = (TrainX-repmat(meanTX,1,mt))./repmat(stdTX,1,mt);
TrainY = (TrainY-repmat(meanTY,1,mt))./repmat(stdTY,1,mt);
DataX = (DataX-repmat(meanTX,1,md))./repmat(stdTX,1,md);
DataY = (DataY-repmat(meanTY,1,md))./repmat(stdTY,1,md);

% calculation of kernel matrix
%disp('kernel matrix XX...');
Kxx = KernelMatrixHIK(TrainX,TrainX,kernel,varargin);
%disp('kernel matrix YY...');
Kyy = KernelMatrixHIK(TrainY,TrainY,kernel,varargin);
%disp('kernel matrix XY...');
Kxy = KernelMatrixHIK(TrainX,TrainY,kernel,varargin);
%disp('kernel matrix YX...');
Kyx = Kxy';

% centeralization of kernel matrix
%disp('Centeralize kernel matrix...');
Kxx_c = KernelCenteralize(Kxx);
Kyy_c = KernelCenteralize(Kyy);
Kxy_c = KernelCenteralize(Kxy);
Kyx_c = Kxy_c';

% clear Kxx Kyy Kxy Kyx

% symmetry of kernel matrix
%disp('Symmetry...');
Kxx_c = (Kxx_c+Kxx_c')/2;
Kyy_c = (Kyy_c+Kyy_c')/2;

% calculation of SFA learning
%disp('slow feature analysis learning...');
A = [(Kxx_c-Kxy_c);(Kyx_c-Kyy_c)]*[Kxx_c-Kyx_c,Kxy_c-Kyy_c]/mt;
B = [Kxx_c,Kxy_c;Kyx_c,Kyy_c]*[Kxx_c,Kxy_c;Kyx_c,Kyy_c]/2/mt;

% regularization
lamda = 1e-5*mean(diag(A));
% lamda = 1;
% B = B+lamda*eye(2*mt);
A = A+lamda*eye(2*mt);

% generalized eigenvalue solution
%disp('eigenvalue problem solution...');
% [V_cita,D] = eig(A,B);
% [V_cita,D] = eig(B,A);
[V_cita,D] = eigs(B,A,20);
% ori_D = diag(D);
D = abs(D);
[eigenv,index] = sort(diag(D),'descend');
V_cita = V_cita(:,index);

% non-zero eigenvalue and eigenvector
% threshold = 1e-8;
% [index] = find(eigenv<threshold);
% eigenv = eigenv(index);
% V_cita = V_cita(:,index);

% eigenvector normalization
%disp('eigenvector normalization...');
aux1 = V_cita'*[Kxx_c,Kxy_c;Kyx_c,Kyy_c]*[Kxx_c,Kxy_c;Kyx_c,Kyy_c]*V_cita/2/mt;
aux2 = 1./sqrt(diag(aux1));
aux3 = repmat(aux2',2*mt,1);
V_cita = V_cita.*aux3;

% projection matrix
%disp('kernel matrix XZx...');
Kxzx = KernelMatrixHIK(TrainX,DataX,kernel,varargin);
%disp('kernel matrix XZy...');
Kxzy = KernelMatrixHIK(TrainX,DataY,kernel,varargin);
%disp('kernel matrix YZx...');
Kyzx = KernelMatrixHIK(TrainY,DataX,kernel,varargin);
%disp('kernel matrix YZx...');
Kyzy = KernelMatrixHIK(TrainY,DataY,kernel,varargin);

% projection method 1
%disp('slow feature projection...');
% Kxzx_c = KernelProjectionCenteralize(Kxx, Kxzx);
% Kyzx_c = KernelProjectionCenteralize(Kyy, Kyzx);
% Kxzy_c = KernelProjectionCenteralize(Kxx, Kxzy);
% Kyzy_c = KernelProjectionCenteralize(Kyy, Kyzy);
% Xsf = V_cita'*[Kxzx_c;Kyzx_c];
% Ysf = V_cita'*[Kxzy_c;Kyzy_c];
% SFeatures = Xsf-Ysf;

% projection method 2
%disp('slow feature projection...');
% K_alpha = (Kxzx-ones(mt,mt)/mt*Kxzx)-(Kxzy-ones(mt,mt)/mt*Kxzy);
% K_beta = (Kyzx-ones(mt,mt)/mt*Kyzx)-(Kyzy-ones(mt,mt)/mt*Kyzy);
% SFeatures = V_cita'*[K_alpha;K_beta];
% Xsf = [];
% Ysf = [];

% projection method 3
%disp('slow feature projection...');
Kxzx_c = Kxzx-ones(mt,mt)/mt*Kxzx-Kxx*ones(mt,md)/mt+ones(mt,mt)/mt*Kxx*ones(mt,md)/mt;
Kyzx_c = Kyzx-ones(mt,mt)/mt*Kyzx-Kyx*ones(mt,md)/mt+ones(mt,mt)/mt*Kyx*ones(mt,md)/mt;
Kxzy_c = Kxzy-ones(mt,mt)/mt*Kxzy-Kxy*ones(mt,md)/mt+ones(mt,mt)/mt*Kxy*ones(mt,md)/mt;
Kyzy_c = Kyzy-ones(mt,mt)/mt*Kyzy-Kyy*ones(mt,md)/mt+ones(mt,mt)/mt*Kyy*ones(mt,md)/mt;
K_alpha = Kxzx_c-Kxzy_c;
K_beta = Kyzx_c-Kyzy_c;
SFeatures = V_cita'*[K_alpha;K_beta];

end









