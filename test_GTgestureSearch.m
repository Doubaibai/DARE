%This script tests GT annotation for deception detection, for Ablation study or independent study
function comb_probs = test_GTgestureSearch(part, method)
addpath('../../../toolbox/libsvm/matlab');
csv_path = '../dataset_trial/gt_anno.mat';
load(csv_path);

num_ges = 5;
ges_id = [5 6 19 21 22];

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




gesture_label = cell(1, num_ges);
trainfile = ['../Scripts_by_ID/trainVideo',num2str(part),'.txt'];
testfile = ['../Scripts_by_ID/testVideo',num2str(part),'.txt'];
fid = fopen(trainfile);
C = textscan(fid, '%s');
fclose(fid);
num_v = length(C{1})/2;

for ges_i = 1:num_ges
           %fea_mat = zeros(num_v, 24576*2);
    lab_vec = zeros(num_v, 1);
    for i = 1:num_v
        [pathstr,name,ext] = fileparts(C{1}{2*i-1});
        videoname = [name,ext];
        eval(sprintf('lab_vec(i)=str2num(fea_dict.%s(ges_id(%d)));',name, ges_i));
    end
    gesture_label{ges_i} = lab_vec;
end
train_score = cell2mat(gesture_label);
train_lab = zeros(num_v,1);
for i = 1:num_v
    train_lab(i) = str2num(C{1}{2*i});
end

%Load test score and labels
fid = fopen(testfile);
C = textscan(fid, '%s');
fclose(fid);
num_v = length(C{1})/2;
gesture_label = cell(1, num_ges);
for ges_i = 1:num_ges
           %fea_mat = zeros(num_v, 24576*2);
    lab_vec = zeros(num_v, 1);
    for i = 1:num_v
        [pathstr,name,ext] = fileparts(C{1}{2*i-1});
        videoname = [name,ext];
        eval(sprintf('lab_vec(i)=str2num(fea_dict.%s(ges_id(%d)));',name,ges_i));
    end
    gesture_label{ges_i} = lab_vec;
end
test_gt_score = cell2mat(gesture_label);
test_lab = zeros(num_v,1);
for i = 1:num_v
    test_lab(i) = str2num(C{1}{2*i});
end


comb_probs = cell(1, length(w));
for comb_i = 1:length(w)
    fprintf('Computing %d out of %d\n', comb_i, length(w));

    tmptrain_fea = train_score.* repmat(w{comb_i},size(train_score,1), 1);
    tmptest_fea = test_gt_score.* repmat(w{comb_i},size(test_gt_score,1), 1);
    
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
        if size(tmptrain_fea(:,flag),2) > 0
            O1 = fitNaiveBayes(tmptrain_fea(:,flag), train_lab);
            C1 = posterior(O1, tmptest_fea(:,flag));
            prob = C1(:,1); 
        else
            prob = zeros(size(tmptest_fea,1),1);
        end
    case 'log'
        B = glmfit(tmptrain_fea, [train_lab ones(size(train_lab,1),1)], 'binomial', 'link', 'logit')
        Z = repmat(B(1), size(test_lab,1),1) + tmptest_fea*B(2:end);
        prob = 1 ./ (1 + exp(-Z));
        prob = 1-prob;
    case 'boost'
        ens = fitensemble(tmptrain_fea,train_lab,'AdaBoostM1',100,'Tree')
        [~, prob] = predict(ens,tmptest_fea)
        prob = prob(:,1);
    case 'linearsvm'
        model = svmtrain(train_lab, tmptrain_fea, '-t 0 -q');
        [pred_gt, acc_gt, prob_gt] = svmpredict(test_lab, tmptest_fea, model);
        lie_id = find(prob_gt<0);
        isign = 1;
        if ~isempty(lie_id)
            if pred_gt(lie_id(1)) == 0
                isign = -1;
            else
                isign = 1;
            end
        else
            if pred_gt(1) == 1
                isign = -1;
            else
                isgin = 1;
            end
        end
        prob = prob_gt*isign;
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
    comb_probs{comb_i} = prob;
end
%[~,~,~,AUC] = perfcurve(test_lab,prob_gt,0)
%tt = sum(pred_gt(test_lab==1)==1);
%tf = sum(pred_gt(test_lab==1)==0);
%ft = sum(pred_gt(test_lab==0)==1);
%ff = sum(pred_gt(test_lab==0)==0);
%fprintf('Partation:%d\tGT_score:\tacc is %f, auc is %f, tt:%d, tf:%d, ft:%d, ff:%d\n',cv, acc_gt(1),AUC,tt,tf,ft,ff);




