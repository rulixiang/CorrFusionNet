function [oa_t1, oa_t2, oa_bi, oa_tr] = Accuracy(pred_t1, pred_t2, label_t1, label_t2)
pred_t1 = reshape(pred_t1, [], 1);
pred_t2 = reshape(pred_t2, [], 1);
label_t1 = reshape(label_t1, [], 1);
label_t2 = reshape(label_t2, [], 1);

num = double(max(label_t1+1));

oa_t1 = sum(pred_t1==label_t1)/length(pred_t1);
oa_t2 = sum(pred_t2==label_t2)/length(pred_t2);
oa_bi = sum((pred_t1==pred_t2)==(label_t1==label_t2))/length(pred_t1);
oa_tr = sum((pred_t1*num+pred_t2)==(label_t1*num+label_t2))/length(pred_t1);


end