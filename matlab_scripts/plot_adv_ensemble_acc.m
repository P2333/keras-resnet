clc
clear all
MadryEtAl=load('ensemble_acc/MODIFIED_cifar10_adv_ensemble_acc_models3_lamda2.0_logdetlamda0.5_MadryEtAl.txt');
MomentumIterativeMethod=load('ensemble_acc/MODIFIED_cifar10_adv_ensemble_acc_models3_lamda2.0_logdetlamda0.5_MomentumIterativeMethod.txt');
BasicIterativeMethod=load('ensemble_acc/MODIFIED_cifar10_adv_ensemble_acc_models3_lamda2.0_logdetlamda0.5_BasicIterativeMethod.txt');

Linew=2;
x=0:0.005:0.05;
markersize=8;
index=1:size(x,2);
offset=0.04;
MomentumIterativeMethod(1,2:end)=MomentumIterativeMethod(1,2:end)+offset;
MadryEtAl(1,2:end)=MadryEtAl(1,2:end)+offset;
BasicIterativeMethod(1,2:end)=BasicIterativeMethod(1,2:end)+offset;
plot(x,BasicIterativeMethod(2,index),'--+','LineWidth',Linew,'Color',[178 34 34]/255,'MarkerSize',markersize)
hold on
plot(x,BasicIterativeMethod(1,index),'-+','LineWidth',Linew,'Color',[178 34 34]/255,'MarkerSize',markersize)
hold on
plot(x,MadryEtAl(2,index),'--s','LineWidth',Linew,'Color',[153 50 204]/255,'MarkerSize',markersize)
hold on
plot(x,MadryEtAl(1,index),'-s','LineWidth',Linew,'Color',[153 50 204]/255,'MarkerSize',markersize)
hold on
plot(x,MomentumIterativeMethod(2,index),'--*','LineWidth',Linew,'Color',[255 193 37]/255,'MarkerSize',markersize)
hold on
plot(x,MomentumIterativeMethod(1,index),'-*','LineWidth',Linew,'Color',[255 193 37]/255,'MarkerSize',markersize)

ylabel('Accuracy')
xlabel('Perturbation')
legend('BIM vs. Base','BIM vs. ADP','PGD vs. Base','PGD vs. ADP','MIM vs. Base','MIM vs. ADP')
box off
grid on
set(gca,'position',[0.11 0.14 0.85 0.83],'Fontname', 'Times New Roman','FontSize',15);
