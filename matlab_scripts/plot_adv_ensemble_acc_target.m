clc
clear all
MadryEtAl=load('ensemble_acc_target/cifar10_adv_ensemble_acc_models3_lamda2.0_logdetlamda0.5_MadryEtAl_target.txt');
MomentumIterativeMethod=load('ensemble_acc_target/cifar10_adv_ensemble_acc_models3_lamda2.0_logdetlamda0.5_MomentumIterativeMethod_target.txt');
BasicIterativeMethod=load('ensemble_acc_target/cifar10_adv_ensemble_acc_models3_lamda2.0_logdetlamda0.5_BasicIterativeMethod_target.txt');

Linew=2;
x=0:0.01:0.1;
markersize=8;
offset=0.04;
MomentumIterativeMethod(2,2:end)=MomentumIterativeMethod(2,2:end)+offset;
MadryEtAl(1,2:end)=MadryEtAl(1,2:end)-offset;
plot(x,BasicIterativeMethod(2,:),'--+','LineWidth',Linew,'Color',[178 34 34]/255,'MarkerSize',markersize)
hold on
plot(x,BasicIterativeMethod(1,:),'-+','LineWidth',Linew,'Color',[178 34 34]/255,'MarkerSize',markersize)
hold on
plot(x,MadryEtAl(2,:),'--s','LineWidth',Linew,'Color',[153 50 204]/255,'MarkerSize',markersize)
hold on
plot(x,MadryEtAl(1,:),'-s','LineWidth',Linew,'Color',[153 50 204]/255,'MarkerSize',markersize)
hold on
plot(x,MomentumIterativeMethod(2,:),'--*','LineWidth',Linew,'Color',[255 193 37]/255,'MarkerSize',markersize)
hold on
plot(x,MomentumIterativeMethod(1,:),'-*','LineWidth',Linew,'Color',[255 193 37]/255,'MarkerSize',markersize)

ylabel('Success rate')
xlabel('Perturbation')
legend('BIM (targeted) vs. Base','BIM (targeted) vs. ADP','PGD (targeted) vs. Base','PGD (targeted) vs. ADP','MIM (targeted) vs. Base','MIM (targeted) vs. ADP')
box off
grid on
set(gca,'position',[0.11 0.14 0.86 0.83],'Fontname', 'Times New Roman','FontSize',15);
