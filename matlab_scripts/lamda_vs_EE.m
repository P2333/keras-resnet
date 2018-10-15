clc
clear all
L=10;
K=3;
Ey=0.6;


lamda=(K./Ey)./log((Ey*(L-1))./(1-Ey));

plot(Ey,lamda)