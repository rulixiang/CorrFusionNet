function [acc_svm, acc_ksfa, val_pred_t1, val_pred_t2, val_bayes_t1, val_bayes_t2] = getAccuracy(val_hist_t1,val_hist_t2,trn_hist_t1,trn_hist_t2,val_prob_t1, val_prob_t2, val_label_t1, val_label_t2)

[SFeatures,V_cita,~] = KernelSFA(val_hist_t1',val_hist_t2',trn_hist_t1',trn_hist_t2','RBF',5);

dim = size(SFeatures, 1);
num = size(SFeatures,2);
varSF = var(SFeatures, 0, 2);
sumSF = sum(SFeatures.^2./repmat(varSF, 1, num),1);
prob = chi2cdf(sumSF,dim);

%% validation svm
[~, val_pred_t1] = max(val_prob_t1');
val_pred_t1 = val_pred_t1-min(val_pred_t1);
[~, val_pred_t2] = max(val_prob_t2');
val_pred_t2 = val_pred_t2-min(val_pred_t2);
[oa_t1, oa_t2, oa_bi, oa_tr] = Accuracy(val_pred_t1, val_pred_t2, val_label_t1, val_label_t2);
acc_svm = [oa_t1, oa_t2, oa_bi, oa_tr];

%% validation ksfa
[val_bayes_t1,val_bayes_t2]=Bayes(val_prob_t1, val_prob_t2,prob);
val_bayes_t1 = val_bayes_t1-min(val_bayes_t1);
val_bayes_t2 = val_bayes_t2-min(val_bayes_t2);
[oa_t1, oa_t2, oa_bi, oa_tr] = Accuracy(val_bayes_t1, val_bayes_t2, val_label_t1, val_label_t2);
acc_ksfa = [oa_t1, oa_t2, oa_bi, oa_tr];

end

