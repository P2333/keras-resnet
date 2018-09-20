clc
clear all

%Setting parameters
select_class=7;
num_class=10;
num_test_all=10000;
num_test=10000;
index=randsample(num_test_all,num_test);

%Load results
%predictions=load('test_predictions_models3_lamda2.0_nonMElamda1.0_epoch180.txt');
predictions=load('test_predictions_models3_lamda2.0_nonMElamda0.0_epoch187.txt');
%predictions=load('test_predictions_models3_lamda2.0_nonMElamda0.5_epoch155.txt');

labels=load('test_labels.txt');

%Sample a subset of results
predictions=predictions(index,:);
labels=labels(index,:);

%Choose samples with selected label
label_num=zeros(num_test,1);
for i=1:num_test
    label_num(i)=find(labels(i,:)==1);
end
select_index=find(label_num==select_class);
S=size(select_index,1);
label_model=[ones(S,1);2*ones(S,1);3*ones(S,1)];

%Get non-maximal predicitons
pred_1=predictions(select_index,1:num_class);
pred_2=predictions(select_index,1+num_class:2*num_class);
pred_3=predictions(select_index,1+2*num_class:3*num_class);

%Get non-maximal prediction
non_pred_1=zeros(S,num_class-1);
non_pred_2=zeros(S,num_class-1);
non_pred_3=zeros(S,num_class-1);
for i=1:S
    non_pred_1(i,:)=pred_1(i,find(labels(i,:)==0))./sum(pred_1(i,find(labels(i,:)==0)),2);
    non_pred_2(i,:)=pred_2(i,find(labels(i,:)==0))./sum(pred_2(i,find(labels(i,:)==0)),2);
    non_pred_3(i,:)=pred_3(i,find(labels(i,:)==0))./sum(pred_3(i,find(labels(i,:)==0)),2);
end





non_pred_all=[non_pred_1;non_pred_2;non_pred_3];

mappedX=tsne(non_pred_all,label_model,2,9,30);
%scatter3(mappedX(:,1), mappedX(:,2), mappedX(:,3));