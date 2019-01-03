clc
clear all

num_all=10000;
num_use=1000;

%fea=load('tsne_features/tsnefeatures_models1of3_lamda0.0_logdetlamda0.0_epoch139.txt');
fea=load('tsne_features/tsnefeatures_models1of3_lamda2.0_logdetlamda0.5_epoch159.txt');

label_name={'airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'};
labels=load('tsne_features/test_labels.txt');
labels_index=zeros(num_all,1);
for i=1:num_all
    labels_index(i)=find(labels(i,:)==1);
end
mappedX=tsne(fea(1:num_use,:), labels_index(1:num_use), 2, 64, 30);
h=gscatter(mappedX(:,1), mappedX(:,2), labels_index(1:num_use));
legend off
box off
axis off