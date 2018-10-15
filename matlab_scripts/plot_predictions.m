clc
clear all

%Setting parameters
num_class=10;
num_test_all=10000;
num_test=1000;
index=randsample(num_test_all,num_test);

%Load results
%predictions=load('test_predictions_models2_lamda2.0_nonMElamda0.0_epoch143.txt');
%predictions=load('test_predictions_models2_lamda2.0_nonMElamda0.3_epoch183.txt');
%predictions=load('test_predictions_models2_lamda2.0_nonMElamda0.5_epoch150.txt');
%predictions=load('test_predictions_models2_lamda2.0_nonMElamda0.7_epoch172.txt');
%predictions=load('test_predictions_models2_lamda2.0_nonMElamda0.9_epoch160.txt');
%predictions=load('test_predictions_models3_lamda2.0_nonMElamda0.0_epoch187.txt');
%predictions=load('test_predictions_models3_lamda2.0_nonMElamda0.5_epoch155.txt');
%predictions=load('test_predictions_models3_lamda2.0_nonMElamda1.0_epoch180.txt');
%predictions=load('test_predictions_models2_lamda2.0_logdetlamda0.3_epoch153.txt');
%predictions=load('test_predictions_models2_lamda2.0_logdetlamda0.7_epoch194.txt');
%predictions=load('test_predictions_models4_lamda2.0_logdetlamda0.3_epoch170.txt');
predictions=load('test_predictions_models3_lamda2.0_logdetlamda0.3_epoch180.txt');


labels=load('test_labels.txt');

%Sample a subset of results
predictions=predictions(index,:);
labels=labels(index,:);

%Get non-maximal predicitons
R_labels=ones(num_test,num_class)-labels;
non_pred_1=predictions(:,1:num_class).*R_labels;
non_pred_2=predictions(:,1+num_class:2*num_class).*R_labels;

%Normalize each non-maximal vector with its l1 norm
x_1=1-sum(non_pred_1,2);
y_1=zeros(num_test,1);

L2norm_1=sqrt(sum(non_pred_1.^2,2));
L2norm_2=sqrt(sum(non_pred_2.^2,2));
Dot_12=sum(non_pred_1.*non_pred_2,2);
cos_12=Dot_12./(L2norm_1.*L2norm_2);
sin_12=sqrt(1-cos_12.^2);

L1norm_2=1-sum(non_pred_1,2);
x_2=L1norm_2.*cos_12;
y_2=L1norm_2.*sin_12;

scatter(x_1,y_1,'*')
hold on
scatter(x_2,y_2,'x')
axis([0 1 0 1])

