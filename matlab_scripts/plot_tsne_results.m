clc
clear all

num_all=10000;
num_use=1000;

fea=load('tsne_features/tsnefeatures_models3of3_lamda0.0_logdetlamda0.0_epoch139.txt');
labels=load('tsne_features/test_labels.txt');
labels_index=zeros(num_all,1);
for i=1:num_all
    labels_index(i)=find(labels(i,:)==1);
end
mappedX=tsne(fea(1:num_use,:), labels_index(1:num_use), 2, 30, 30);
gscatter(mappedX(:,1), mappedX(:,2), labels_index(1:num_use));