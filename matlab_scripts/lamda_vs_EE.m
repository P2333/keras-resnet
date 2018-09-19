clc
clear all
L=10;
K=2;
Ey=0.15;


lamda=(K./Ey)./log((Ey*(L-1))./(1-Ey));

plot(Ey,lamda)