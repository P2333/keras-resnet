clc
clear all

%Setting parameters
num_class=10;
num_test_all=10000;



%Load results
predictions_1=load('test_predictions_models3_lamda0.0_logdetlamda0.0_epoch139.txt');
predictions_2=load('test_predictions_models3_lamda2.0_logdetlamda0.0_epoch157.txt');
predictions_3=load('test_predictions_models3_lamda2.0_logdetlamda0.3_epoch180.txt');


labels=load('test_labels.txt');


%Get non-maximal predicitons
predictions=predictions_3;
pred_1=predictions(:,1:num_class);
pred_2=predictions(:,1+num_class:2*num_class);
pred_3=predictions(:,1+2*num_class:3*num_class);