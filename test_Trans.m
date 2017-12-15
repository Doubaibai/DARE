% This script test under 10-fold Cross Validation, divided by ID, how dtfv works
% Using transcript feature
function prob = test_Trans(part, method)
run('../../../toolbox/vlfeat-0.9.20/toolbox/vl_setup')
addpath('../../../toolbox/libsvm/matlab');
%part = 1;
trainfile = ['../Scripts_by_ID/trainVideo',num2str(part),'.txt'];
testfile = ['../Scripts_by_ID/testVideo',num2str(part),'.txt'];
feapath = '../dataset_trial/TranFeats/';

%Build dictionary
if ~exist('Tran_words.mat', 'file')
    traindic = ['../Scripts_by_ID/trainVideo0.txt'];
    testdic = ['../Scripts_by_ID/testVideo0.txt'];

    fid = fopen(traindic);
    Ctr = textscan(fid, '%s');
    fclose(fid);
    fid = fopen(testdic);
    Cte = textscan(fid, '%s');
    fclose(fid);

    data = [];
    num_tr = length(Ctr{1})/2;
    for i = 1:num_tr
        [pathstr,name,ext] = fileparts(Ctr{1}{2*i-1});
        feafile = fullfile(feapath, [name, '.txt.npy.mat']);
        load(feafile);
        data = [data; reshape(A, size(A,2), size(A,3))];
    end
    num_te = length(Cte{1})/2;
    for i = 1:num_te
        [pathstr,name,ext] = fileparts(Cte{1}{2*i-1});
        feafile = fullfile(feapath, [name, '.txt.npy.mat']);
        load(feafile);
        data = [data; reshape(A, size(A,2), size(A,3))];
    end
    fprintf('Total word number is %d.\n', size(data,1));
    save('Tran_words.mat', 'data');
    else
        load('Tran_words.mat');
        fprintf('Load words finished.\n');
    end
    % Dictionary
    numClusters = 64;
    if ~exist('Tran_dict.mat', 'file')
    fprintf('Start building dictionary.\n');
    data = data';
    [means, covariances, priors] = vl_gmm(data, numClusters);
    fprintf('Build dictionary finished.\n');
    save('Tran_dict.mat','means', 'covariances', 'priors');
else
    load('Tran_dict.mat');
end
% Load data
fid = fopen(trainfile);
C = textscan(fid, '%s');
fclose(fid);
fea_dim = 2*300*numClusters;%+24576+27648;

num_v = length(C{1})/2;
train_fea = zeros(num_v, fea_dim);
train_lab = zeros(num_v, 1);
for i = 1:num_v
    [pathstr,name,ext] = fileparts(C{1}{2*i-1});
    s = findstr(name, 'lie');
    lab = isempty(s);
    feafile = fullfile(feapath, [name, '.txt.npy.mat']);
    load(feafile);
    tmpdata = reshape(A, size(A,2), size(A,3))';
    encoding = vl_fisher(tmpdata, means, covariances, priors);
    train_fea(i,:) = encoding';
    train_lab(i) = lab;
end

%% test phase
fid = fopen(testfile);
C = textscan(fid, '%s');
fclose(fid);
num_v = length(C{1})/2;
test_fea = zeros(num_v, fea_dim);
test_lab = zeros(num_v, 1);
for i = 1:num_v
    [pathstr,name,ext] = fileparts(C{1}{2*i-1});
    s = findstr(name, 'lie');
    lab = isempty(s);
    feafile = fullfile(feapath, [name, '.txt.npy.mat']);
    load(feafile);
    tmpdata = reshape(A, size(A,2), size(A,3))';
    encoding = vl_fisher(tmpdata, means, covariances, priors);
    test_fea(i,:) = encoding';
    test_lab(i) = lab;
end

switch(method)
case 'NN'
    net = feedforwardnet(10);
    net.trainFcn = 'trainscg';
    net = configure(net, train_fea', train_lab');
    net = train(net, train_fea', train_lab');
    prob = net(test_fea');
case 'tree'
    tc = fitctree(train_fea, train_lab);
    [label,score,node,cnum] = predict(tc, test_fea);
    prob = score(:,1);
case 'randforest'
    BaggedEnsemble = TreeBagger(50,train_fea,train_lab,'OOBPred','On');
    [label,scores] = predict(BaggedEnsemble, test_fea);
    prob = scores(:,1);
case 'bayes'
    flag = bitand(var(train_fea(train_lab==1,:))>1e-10,var(train_fea(train_lab==0,:))>1e-10); %clear 0 variance features
    O1 = fitNaiveBayes(train_fea(:,flag), train_lab);
    C1 = posterior(O1, test_fea(:,flag));
    prob = C1(:,1);
case 'log'
    B = glmfit(train_fea, [train_lab ones(size(train_lab,1),1)], 'binomial', 'link', 'logit');
    Z = repmat(B(1), size(test_lab,1),1) + test_fea*B(2:end);
    prob = 1 ./ (1 + exp(-Z));
    prob = 1-prob;
case 'boost'
    ens = fitensemble(train_fea,train_lab,'AdaBoostM1',100,'Tree')
    [~, prob] = predict(ens,test_fea)
    prob = prob(:,1);
case 'linearsvm'
    %model = svmtrain(train_lab, tmptrain_fea, '-t 0 -q -b 1');
    %fprintf('Finished training.\n');
    %[pred, acc, prob] = svmpredict(test_lab, tmptest_fea, model, '-q -b 1');
    %prob = prob(:,2);
    model = svmtrain(train_lab, train_fea, '-t 0 -q');
    [pred, acc, prob] = svmpredict(test_lab, test_fea, model, '-q');

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
    model = svmtrain(train_lab, train_fea, '-t 1 -c 1 -g 1 -q');
    [pred, acc, prob] = svmpredict(test_lab, test_fea, model, '-q');

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

