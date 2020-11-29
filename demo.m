
clear;clc

%% load data
% subset 1 for training, subset 5 for testing
imgsize = [48 42];
load data\YB_subset1.mat
trdat = [];
trls = train_label;
for i=1:size(train_data, 2)
    t = reshape(train_data(:,i), [192 168]);
    t = imresize(t, imgsize);
    t = t(:);
    trdat = [trdat t];
end
clear train_data train_label

load data\YB_subset5.mat
ttdat = train_data;
ttls = train_label;
ttdat = [];
for i=1:size(train_data, 2)
    t = reshape(train_data(:,i), [192 168]);
    t = imresize(t, imgsize);
    t = t(:);
    ttdat = [ttdat t];
end
clear train_data train_label
%% parameter set
options = [];
options.gamma1 =  3;
options.gamma2 =  3;
options.gamma3 =  3;
options.alpha  =  1;   % tune the parameter according to your dataset
options.beta   =  0.1;
options.delta  =  1;

%% 
n = size(ttdat, 2);
Pred_label = zeros(1,length(ttls));
for Index = 1:n
    if mod(Index, 100)==0
        fprintf('%d / %d \n', Index, n);
    end
    y = ttdat(:,Index);
    w = ComputeWeight(trdat, trls, y);    
    [x] = EGSNR(trdat, trls, y, w, imgsize, options);
    [pred] = classify(x, trdat, trls, imgsize, options.gamma1);
    Pred_label(Index) = pred;
end
acc = mean(Pred_label(:) == ttls(:));
fprintf('acc: %.2f \n', acc*100);
