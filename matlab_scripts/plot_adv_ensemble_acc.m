clc
clear all
MadryEtAl=load('cifar10_adv_ensemble_acc_models3_lamda2.0_logdetlamda0.3_MadryEtAl.txt');
MomentumIterativeMethod=load('cifar10_adv_ensemble_acc_models3_lamda2.0_logdetlamda0.3_MomentumIterativeMethod.txt');

Linew=2;
x=0:0.01:0.1;
markersize=8;

plot(x,MadryEtAl(1,:),'-s','LineWidth',Linew,'Color',[153 50 204]/255,'MarkerSize',markersize)
hold on
plot(x,MadryEtAl(2,:),'--s','LineWidth',Linew,'Color',[153 50 204]/255,'MarkerSize',markersize)
hold on
plot(x,MomentumIterativeMethod(1,:),'-*','LineWidth',Linew,'Color',[255 193 37]/255,'MarkerSize',markersize)
hold on
plot(x,MomentumIterativeMethod(2,:),'--*','LineWidth',Linew,'Color',[255 193 37]/255,'MarkerSize',markersize)
ylabel('Accuracy')
xlabel('Eps')
box off
grid on