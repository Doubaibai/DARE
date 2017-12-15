% Test split for gesture labels, for video chunks data
% run in directory of dtfv_learning
addpath('other');
addpath('../../../toolbox/libsvm/matlab');
load('../dataset_trial/saved_anno.mat');
load('../dataset_trial/frame_rates.mat');

% Build feature Map of easy load of FV feats
FV_path = '../dataset_trial/FVs_chunks';
CNN_path = '../dataset_trial/CNNfeats';

FVfeaMap = FV_feaMap(FV_path);
CNNfeaMap = CNN_feaMap(CNN_path);

gesture_score = cell(1,10);
gesture_acc = cell(1,10);
gesture_pred = cell(1,10);
train_lab = cell(1, 10);
test_lab = cell(1, 10);
gesture_scorei_train = cell(1,10);
num_chunks_train = cell(1,10);
num_chunks_test = cell(1,10);

num_ges = 5;

use_CNN = 1;
for cv = 0:9
    gesture_score{cv+1} = cell(1, num_ges);
    gesture_acc{cv+1} = cell(1, num_ges);
    gesture_pred{cv+1} = cell(1, num_ges);
    train_lab{cv+1} = cell(1, num_ges);
    test_lab{cv+1} = cell(1, num_ges);

    total_posnum_tr = zeros(1,5); % number of total pos for training
    total_negnum_tr = zeros(1,5); % number of total neg for training
    train_au_lab = []; % train facial unit labels
    test_au_lab = [];
    current_i = 1;  % used for accumulating id of each sample
    fprintf('CV: %d\n', cv);
    
    %% Loading data
    % Train data 
    trainfile = ['../Scripts_by_ID/trainVideo',num2str(cv),'.txt'];
    testfile = ['../Scripts_by_ID/testVideo',num2str(cv),'.txt'];
    fid = fopen(trainfile);
    C = textscan(fid, '%s');
    fclose(fid);
    num_v = length(C{1})/2;
    feat_train_CNN = [];
    feat_train_FV = [];
    num_chunks_train{cv+1} = zeros(num_v,1);
    for i = 1:num_v
        [pathstr,name,ext] = fileparts(C{1}{2*i-1});
        disp(name)
        chunk_names = get_chunkname(name);
        num_chunks_train{cv+1}(i) = length(chunk_names);
        for chunk_id = 1: length(chunk_names)
            % check whether is a bad chunk
            if ~bad_chunk(chunk_names{chunk_id})
                eval(sprintf('chunk_label=fea_dict.%s;',chunk_names{chunk_id}));
                chunk_name = chunk_names{chunk_id};
                chunk_label = str2num(chunk_label);
                train_au_lab = [train_au_lab; chunk_label'];  % #sample x #AU (5)
                % get features CNN
                CNN_feats = get_CNNfeats(chunk_name, CNNfeaMap, frame_rates);
                %CNN_feats = mean(CNN_feats, 1);  % 1-by-4096
                % get features FV
                FV_feats = get_FVfeats(chunk_name, FVfeaMap);
                % update step
                feat_train_CNN(current_i,:) = CNN_feats;
                feat_train_FV(current_i,:) = FV_feats;
                current_i = current_i +1;
            end
        end
    end
    disp('train#')
    sum(train_au_lab, 1)

    % Test data
    fid = fopen(testfile);
    C = textscan(fid, '%s');
    fclose(fid);
    num_v = length(C{1})/2;
    feat_test_CNN = [];
    feat_test_FV = [];
    current_i = 1;
    num_chunks_test{cv+1} = zeros(num_v,1);
    for i = 1:num_v
        [pathstr,name,ext] = fileparts(C{1}{2*i-1});
        chunk_names = get_chunkname(name);
        num_chunks_test{cv+1}(i) = length(chunk_names);
        for chunk_id = 1: length(chunk_names)
            % check whether is a bad chunk
            if ~bad_chunk(chunk_names{chunk_id})
                eval(sprintf('chunk_label=fea_dict.%s;',chunk_names{chunk_id}));
                chunk_name = chunk_names{chunk_id};
                chunk_label = str2num(chunk_label);
                test_au_lab = [test_au_lab; chunk_label'];  % #sample x #AU (5)
                % get features CNN
                CNN_feats = get_CNNfeats(chunk_name, CNNfeaMap, frame_rates);
                %CNN_feats = mean(CNN_feats, 1);  % 1-by-4096
                % get features FV
                FV_feats = get_FVfeats(chunk_name, FVfeaMap);
                % update step
                feat_test_CNN(current_i,:) = CNN_feats;
                feat_test_FV(current_i,:) = FV_feats;
                current_i = current_i +1;
            end
        end
    end
    disp('test#')
    sum(test_au_lab, 1)
    
    %% Training classifiers
    for ges_i = 1:num_ges
        fprintf('CV: %d, ges_id: %d\n', cv, ges_i);
        lab_vec = train_au_lab(:, ges_i); 
        if use_CNN == 1
            fea_mat = feat_train_CNN;
        else
            fea_mat = feat_train_FV;
        end
        train_lab{cv+1}{ges_i} = lab_vec;
        model = svmtrain(lab_vec, fea_mat, '-t 0 -q -b 1');
        [pred, acc, prob] = svmpredict(lab_vec,fea_mat, model, '-b 1');
        gesture_score_train{cv+1}{ges_i} = prob(:,2);
        % Test
        lab_vec = test_au_lab(:, ges_i);
        if use_CNN == 1
            fea_mat = feat_test_CNN;
        else
            fea_mat = feat_test_FV;
        end
        [pred, acc, prob] = svmpredict(lab_vec,fea_mat, model, '-b 1');
        pred
        gesture_score{cv+1}{ges_i} = prob(:,2);
        gesture_acc{cv+1}{ges_i} = acc;
        gesture_pred{cv+1}{ges_i} = pred;
        test_lab{cv+1}{ges_i} = lab_vec;
    end
    cv_gesture_score = gesture_score{cv+1};
    %save(['CNN_gesture_score',num2str(cv),'.mat'],'cv_gesture_score');

end
if use_CNN == 1
    appname = 'CNN';
else
    appname = 'FV';
end
save(['total_',appname,'_gesture_score.mat'], 'gesture_score');
save(['total_',appname,'_gesture_pred.mat'], 'gesture_pred');
save(['total_',appname,'ges_results.mat'], 'gesture_score', 'gesture_pred', 'train_lab', 'test_lab','gesture_score_train','num_chunks_train','num_chunks_test');
