function  [label_t1,label_t2]=Bayes(pred_prob_t1,pred_prob_t2,prob)
% len=length(pred_label);
label_t1=zeros(size(pred_prob_t1,1),1);
label_t2=zeros(size(pred_prob_t2,1),1);
num=size(pred_prob_t1,2);
for k1=1:length(label_t1)
    matrix=prob(k1)*ones(num)-prob(k1)*eye(num);
    matrix=matrix+(1-prob(k1))*eye(num);
    prob1=pred_prob_t1(k1,:);
    prob2=pred_prob_t2(k1,:);
    for kk1=1:num
        for kk2=1:num
            matrix(kk1,kk2)=matrix(kk1,kk2)*prob1(kk1)*prob2(kk2);
        end
    end
    [~,loc]=max(reshape(matrix,1,[]));
    label_t2(k1)=floor((loc-1)/num)+1;
    label_t1(k1)=mod(loc,num);
    if(label_t1(k1)==0)
        label_t1(k1)=num;
    end
    if(label_t2(k1)==0)
        label_t2(k1)=num;
    end
end
% label=[label_t1;label_t2];
end