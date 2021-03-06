clc
clear all
Tran=load('transfer_acc/cifar10_transfer_acc_models3_lamda2.0_logdetlamda0.5_eps0.04_MomentumIterativeMethod_withensemble_target.txt');
%colormap('summer')
%Tran=load('transfer_acc/cifar10_transfer_acc_models3_lamda2.0_logdetlamda0.5_eps0.04_MadryEtAl_withensemble_target.txt');
index=[1,2,3];
indexb=[5,6,7];

i=indexb;

if i==index
    strs={'Net_1(ADP)', 'Net_2(ADP)', 'Net_3(ADP)'};
else
    strs={'Net_1(Base)', 'Net_2(Base)', 'Net_3(Base)'};
end

Tran=Tran(i,i);
colormap(flipud(gray))

imagesc(Tran,[0 1])
%colorbar
num_model=size(index,2);

textStrings = num2str(Tran(:), '%0.2f');       % Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  % Remove any space padding
[x, y] = meshgrid(1:num_model);  % Create x and y coordinates for the strings
hStrings = text(x(:), y(:), textStrings(:), ...  % Plot the strings
                'HorizontalAlignment', 'center','FontSize',18);
midValue = mean(get(gca, 'CLim'));  % Get the middle value of the color range
textColors = repmat(Tran(:) > midValue, 1, 3);  % Choose white or black for the
                                               %   text color of the strings so
                                               %   they can be easily seen over
                                               %   the background color
set(hStrings, {'Color'}, num2cell(textColors, 2));  % Change the text colors

set(gca, 'XTick', 1:num_model, ...                             % Change the axes tick marks
         'XTickLabel', strs, ...  %   and tick labels
         'YTick', 1:num_model, ...
         'YTickLabel', strs, ...
         'TickLength', [0 0], 'FontName', 'Times New Roman');
