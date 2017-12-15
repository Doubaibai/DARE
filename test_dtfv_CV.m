% This script test under 10-fold Cross Validation, divided by ID, how dtfv works
function prob = test_dtfv_CV(part,feaMap, method)
addpath('../../../toolbox/libsvm/matlab');
%part = 1;
%fea_path = '../dataset_trial/FVs';
trainfile = ['../Scripts_by_ID/trainVideo',num2str(part),'.txt'];
testfile = ['../Scripts_by_ID/testVideo',num2str(part),'.txt'];

fid = fopen(trainfile);
C = textscan(fid, '%s');
fclose(fid);
fea_types = {'mbhx','mbhy'};
fea_dim = 24576*2;%+24576+27648;

num_v = length(C{1})/2;
train_fea = zeros(num_v, fea_dim);
train_lab = zeros(num_v, 1);
for i = 1:num_v
    [pathstr,name,ext] = fileparts(C{1}{2*i-1});
    videoname = [name,ext];
    s = findstr(name, 'lie');
    lab = isempty(s);
    fea = [];
    for j = 1: length(fea_types)
        %fea_file = fullfile(fea_path,[videoname,'.',fea_types{j},'.fv.txt']);
        %disp(fea_file)
        %fea = [fea, importdata(fea_file)];
        fea = [fea, feaMap([videoname,'.',fea_types{j},'.fv.txt'])];
    end
    train_fea(i,:) = fea;
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
    videoname = [name,ext];
    s = findstr(name, 'lie');
    lab = isempty(s);
    fea = [];
    for j = 1: length(fea_types)
        %fea_file = fullfile(fea_path,[videoname,'.',fea_types{j},'.fv.txt']);
        %fea = [fea, importdata(fea_file)];
        fea = [fea, feaMap([videoname,'.',fea_types{j},'.fv.txt'])];
    end
    test_fea(i,:) = fea;
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
    B = glmfit(train_fea, [train_lab ones(size(train_lab,1),1)], 'binomial', 'link', 'logit')
    Z = repmat(B(1), size(test_lab,1),1) + test_fea*B(2:end);
    prob = 1 ./ (1 + exp(-Z));
    prob = 1-prob;
case 'boost'
    ens = fitensemble(train_fea,train_lab,'AdaBoostM1',100,'Tree')
    [~, prob] = predict(ens,test_fea)
    prob = prob(:,1);
case 'linearsvm'
    %model = svmtrain(train_lab, train_fea, '-t 0 -b 1 -q')
    %fprintf('Finished training.\n');
    %[pred, acc, prob] = svmpredict(test_lab,test_fea, model, '-b 1 -q');
    %prob = prob(:,2);

    model = svmtrain(train_lab, train_fea, '-t 0 -q')
    [pred, acc, prob] = svmpredict(test_lab,test_fea, model, ' -q');
    lie_id = find(prob<0); % find score <0 sample
    if ~isempty(lie_id) % if have such sample
        if pred(lie_id(1)) == 0 % if pred of the sample is deceptive, reverse the score
            isign = -1;
        else
            isign = 1;
        end
    else % if all sample scores >0
        if pred(1) == 1 % if pred is truthful, then reverse all scores
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

