% This script test under 10-fold Cross Validation, divided by ID, how predicted gesture from small intervals works. In a grid search way for all gestures.
function [lab_vec, comb_probs, w] = test_gestureSearch(part, method)
% part is from 0 to 9
% abl_id is the ablation gesture id

addpath('../../../toolbox/libsvm/matlab');
addpath('other');
%part = 1;
trainfile = ['../Scripts_by_ID/trainVideo',num2str(part),'.txt'];
testfile = ['../Scripts_by_ID/testVideo',num2str(part),'.txt'];
fid = fopen(trainfile);
C = textscan(fid, '%s');
fclose(fid);

fea_dim = 5;% test 5 dim, other to be all zeros % when fea_dim =1 this is independent feature study
R = load('total_FVges_results.mat');
gesture_train = R.gesture_score_train;
gesture_test = R.gesture_score;

num_v = length(C{1})/2;
num_C = length(gesture_train{part+1}{1});
fea_mat = zeros(num_C, fea_dim);
fea_mat = cell2mat(gesture_train{part+1});
% Do a grid search to find best weights for ges_id
combs = unique(combnk([0 0 0 0 0 .5 .5 .5 .5 .5 1 1 1 1 1], 5), 'rows')
comb_i = 1;
for i = 1:size(combs,1)
    perW = unique(perms(combs(i,:)), 'rows')
    for j = 1:size(perW, 1)
        %tmpw = repmat(perW(j,:), num_C, 1);
        w{comb_i} = perW(j,:);
        comb_i = comb_i +1;
    end
end


lab_vec = [];
current_i = 1;
vid_i = 1;
for i = 1:num_v
    [pathstr,name,ext] = fileparts(C{1}{2*i-1});
    chunk_names = get_chunkname(name);
    if length(chunk_names)>0
        vid_fea = [];  % feature of the video
        for chunk_id = 1:length(chunk_names)
            if ~bad_chunk(chunk_names{chunk_id})
                vid_fea = [vid_fea;fea_mat(current_i,:)];
                current_i = current_i +1;
            end
        end
        videoname = [name,ext];
        s = findstr(name, 'lie');
        lab = isempty(s);
    %fea = [];
    % load feature
    %for j = 1: length(fea_types)
    %    fea_file = fullfile(fea_path,[videoname,'.',fea_types{j},'.fv.txt']);
    %    disp(fea_file)
    %    fea = [fea, importdata(fea_file)];
    %end
    %fea_mat(i,:) = fea;
        train_fea(vid_i, :) = mean(vid_fea, 1);  % average pooling
        %train_fea(vid_i, :) = max(vid_fea, [], 1);  % max pooling 
        lab_vec(vid_i) = lab;
        vid_i = vid_i+1;
    end
end
assert(current_i-1 == size(fea_mat, 1));
train_lab = lab_vec';

%% test phase
fid = fopen(testfile);
C = textscan(fid, '%s');
fclose(fid);
num_v = length(C{1})/2;
num_C = length(gesture_test{part+1}{1});
fea_mat = zeros(num_C, fea_dim);
%gesture_test{part+1}(abl_id)=[];
fea_mat = cell2mat(gesture_test{part+1});
%fea_mat(:,abl_id) = 0; %gesture_test{part+1}{abl_id};
%lab_vec = zeros(num_v, 1);
lab_vec = [];
current_i = 1;
vid_i = 1;
skip_id = []; % record test video ids that not valid
for i = 1:num_v
    [pathstr,name,ext] = fileparts(C{1}{2*i-1});
    chunk_names = get_chunkname(name);
    if length(chunk_names)>0
        vid_fea = [];  % feature of the video
        for chunk_id = 1:length(chunk_names)
            if ~bad_chunk(chunk_names{chunk_id})
                vid_fea = [vid_fea;fea_mat(current_i,:)];
                current_i = current_i +1;
            end
        end
        videoname = [name,ext];
        s = findstr(name, 'lie');
        lab = isempty(s);
        test_fea(i, :) = mean(vid_fea, 1);
        %test_fea(i, :) = max(vid_fea, [], 1); 
        lab_vec(i) = lab;
        %vid_i = vid_i+1;
    else
        skip_id = [skip_id, i];
        test_fea(i, :) = zeros(1, fea_dim);
        lab_vec(i) = 0;
    end
end
assert(current_i-1 == size(fea_mat,1));
test_lab = lab_vec';
 



comb_probs = cell(1, length(w));
for comb_i = 1:length(w)
    fprintf('Computing %d out of %d\n', comb_i, length(w));
    % for each combinatorial weights
    %lab_vec = zeros(num_v, 1);
    tmptrain_fea = train_fea.* repmat(w{comb_i},size(train_fea,1), 1);
    tmptest_fea = test_fea.* repmat(w{comb_i},size(test_fea,1), 1);
    switch(method)
    case 'NN'
        net = feedforwardnet(10);
        net.trainFcn = 'trainscg';
        net = configure(net, tmptrain_fea', train_lab');
        net = train(net, tmptrain_fea', train_lab');
        prob = net(tmptest_fea');
    case 'tree'
        tc = fitctree(tmptrain_fea, train_lab);
        [label,score,node,cnum] = predict(tc, tmptest_fea);
        prob = score(:,1);
    case 'randforest'
        BaggedEnsemble = TreeBagger(50,tmptrain_fea,train_lab,'OOBPred','On');
        [label,scores] = predict(BaggedEnsemble, tmptest_fea);
        prob = scores(:,1);
    case 'bayes'
        flag = bitand(var(tmptrain_fea(train_lab==1,:))>1e-10,var(tmptrain_fea(train_lab==0,:))>1e-10); %clear 0 variance features
        if size(tmptrain_fea(:,flag),2) >0
            O1 = fitNaiveBayes(tmptrain_fea(:,flag), train_lab);
            C1 = posterior(O1, tmptest_fea(:,flag));
            prob = C1(:,1);
        else
            prob = zeros(size(tmptest_fea,1),1);
        end
    case 'log'
        B = glmfit(tmptrain_fea, [train_lab ones(size(train_lab,1),1)], 'binomial', 'link', 'logit');
        Z = repmat(B(1), size(test_lab,1),1) + tmptest_fea*B(2:end);
        prob = 1 ./ (1 + exp(-Z));
        prob = 1-prob;
    case 'boost'
        ens = fitensemble(tmptrain_fea,train_lab,'AdaBoostM1',100,'Tree')
        [~, prob] = predict(ens,tmptest_fea)
        prob = prob(:,1);
    case 'linearsvm'
        %model = svmtrain(train_lab, tmptrain_fea, '-t 0 -q -b 1');
        %fprintf('Finished training.\n');
        %[pred, acc, prob] = svmpredict(test_lab, tmptest_fea, model, '-q -b 1');
        %prob = prob(:,2);
        model = svmtrain(train_lab, tmptrain_fea, '-t 0 -q');
        [pred, acc, prob] = svmpredict(test_lab, tmptest_fea, model, '-q');

        lie_id = find(prob<0);
        if ~isempty(lie_id)
            if pred(lie_id(1)) == 0
                isign = -1;
            else
                isign = 1;
            end
        else
            if pred(1) == 1
                isign = -1;
            else
                isign = 1;
            end
        end
        prob = isign*prob;
    case 'kernelsvm'
        %model = svmtrain(train_lab, tmptrain_fea, '-t 0 -q -b 1');
        %fprintf('Finished training.\n');
        %[pred, acc, prob] = svmpredict(test_lab, tmptest_fea, model, '-q -b 1');
        %prob = prob(:,2);
        model = svmtrain(train_lab, tmptrain_fea, '-t 1 -c 1 -g 1 -q');
        [pred, acc, prob] = svmpredict(test_lab, tmptest_fea, model, '-q');

        lie_id = find(prob<0);
        if ~isempty(lie_id)
            if pred(lie_id(1)) == 0
                isign = -1;
            else
                isign = 1;
            end
        else
            if pred(1) == 1
                isign = -1;
            else
                isign = 1;
            end
        end
        prob = isign*prob;
    end
    prob(skip_id) = nan;
    
           comb_probs{comb_i} = prob;
end
%[~,~,~,AUC] = perfcurve(lab_vec,prob,0)
%tt = sum(pred(lab_vec==1)==1);
%tf = sum(pred(lab_vec==1)==0);
%ft = sum(pred(lab_vec==0)==1);
%ff = sum(pred(lab_vec==0)==0);

%fprintf('Partation:%d\tacc is %f, auc is %f, tt:%d, tf:%d, ft:%d, ff:%d\n',part, acc(1),AUC,tt,tf,ft,ff);

