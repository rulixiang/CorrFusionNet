clear;clc
close all
addpath(genpath('./libsvm-3.22'))
matDir = dir('./results');
load label.mat
val_svm = zeros(10,4);
val_ksfa = zeros(10,4);
tst_svm = zeros(10,4);
tst_ksfa = zeros(10,4);

val_svm_t1 = zeros(10, length(val_label_t1));
val_svm_t2 = zeros(10, length(val_label_t2));
val_ksfa_t1 = zeros(10, length(val_label_t1));
val_ksfa_t2 = zeros(10, length(val_label_t2));
tst_svm_t1 = zeros(10, length(tst_label_t1));
tst_svm_t2 = zeros(10, length(tst_label_t2));
tst_ksfa_t1 = zeros(10, length(tst_label_t1));
tst_ksfa_t2 = zeros(10, length(tst_label_t2));
tic
for k1=3:length(matDir)
    matName = matDir(k1).name;
    load(['./results/', matName])
    %trn_hist_t1 = trn_hist_t1(1:1000,:);
    %trn_hist_t2 = trn_hist_t2(1:1000,:);
    
    [acc_svm, acc_ksfa, val_pred_t1, val_pred_t2, val_bayes_t1, val_bayes_t2] = getAccuracy(val_hist_t1,val_hist_t2,trn_hist_t1,trn_hist_t2,val_prob_t1, val_prob_t2, val_label_t1, val_label_t2);
    val_svm(k1-2,:)=acc_svm;
    val_ksfa(k1-2,:)=acc_ksfa;
    val_svm_t1(k1-2,:)=val_pred_t1;
    val_svm_t2(k1-2,:)=val_pred_t2;
    val_ksfa_t1(k1-2,:)=val_bayes_t1;
    val_ksfa_t2(k1-2,:)=val_bayes_t2;
    
    disp([num2str(k1-2),'-th validation accuracy using SVM...'])
    disp(num2str(acc_svm))
    disp([num2str(k1-2),'-th validation accuracy using KSFA...'])
    disp(num2str(acc_ksfa))
    
    [acc_svm, acc_ksfa, tst_pred_t1, tst_pred_t2, tst_bayes_t1, tst_bayes_t2] = getAccuracy(tst_hist_t1,tst_hist_t2,trn_hist_t1,trn_hist_t2,tst_prob_t1, tst_prob_t2, tst_label_t1, tst_label_t2);
    tst_svm(k1-2,:)=acc_svm;
    tst_ksfa(k1-2,:)=acc_ksfa;
    tst_svm_t1(k1-2,:)=tst_pred_t1;
    tst_svm_t2(k1-2,:)=tst_pred_t2;
    tst_ksfa_t1(k1-2,:)=tst_bayes_t1;
    tst_ksfa_t2(k1-2,:)=tst_bayes_t2;
    
    disp([num2str(k1-2),'-th testing accuracy using SVM...'])
    disp(num2str(acc_svm))
    disp([num2str(k1-2),'-th testing accuracy using KSFA...'])
    disp(num2str(acc_ksfa))
    toc
    
    disp(char(10))
    
end

save acc_rbf.mat val_svm val_ksfa tst_svm tst_ksfa val_svm_t1 val_svm_t2 val_ksfa_t1 val_ksfa_t2 tst_svm_t1 tst_svm_t2 tst_ksfa_t1 tst_ksfa_t2