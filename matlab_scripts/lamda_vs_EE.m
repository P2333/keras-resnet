clc
clear all
L=1000;
K=3;
Ey=0.8;


lamda=(K./Ey)./log((Ey*(L-1))./(1-Ey));

%plot(Ey,lamda)