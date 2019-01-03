clc
clear all
L=100;
K=9;
Ey=0.7;


lamda=(K./Ey)./log((Ey*(L-1))./(1-Ey));

%plot(Ey,lamda)