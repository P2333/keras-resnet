clc
clear all

%Setting parameters
num_class=10;
num_test_all=10000;

labels=load('test_predictions/test_labels.txt');

%Load results
preds=load('test_predictions/test_predictions_models3_lamda2.0_logdetlamda0.5_epoch159.txt');
%preds=load('test_predictions/test_predictions_models3_lamda2.0_logdetlamda0.0_epoch157.txt');
%preds=load('test_predictions/test_predictions_models3_lamda0.0_logdetlamda0.0_epoch139.txt');

preds_1=preds(:,1:num_class);
preds_2=preds(:,1+num_class:2*num_class);
preds_3=preds(:,1+2*num_class:3*num_class);

Vol_all=zeros(num_test_all,1);
for i=1:num_test_all
    p1=preds_1(i,:);
    p2=preds_2(i,:);
    p3=preds_3(i,:);
    l=find(labels(i,:)==1);
    p1(l)=[];
    p2(l)=[];
    p3(l)=[];
    p1=p1/norm(p1);
    p2=p2/norm(p2);
    p3=p3/norm(p3);
    m=[p1;p2;p3];
    Vol_all(i)=norm(sqrt(det(m*m')));
end
mean(Vol_all)
std(Vol_all)
histogram(Vol_all)