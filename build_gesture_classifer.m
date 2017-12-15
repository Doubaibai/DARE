%This script builds facial gesture classifer using dtfv feature
addpath('/media/bharat/HDD/zhe/libsvm/matlab');
csv_path = '../Annotation/saved_anno.mat';
load(csv_path);

num_ges = 31;

% read CV
addpath('/media/bharat/HDD/zhe/libsvm/matlab');
%part = 1;
fea_path = '/media/bharat/HDD/zhe/Real-life_Deception_Detection_2016/dtfv_learning/tmp';
fea_types = {'mbhx','mbhy'};

% Load all features
if ~exist('feaMap.mat')
    feaMap = containers.Map;
    for fea_i = 1:length(fea_types)
        fea_ext = [fea_types{fea_i},'.fv.txt'];
        fea_list = dir([fea_path,'/*',fea_ext]);
        for vid_i = 1:length(fea_list)
            tmpfea = importdata(fullfile(fea_path,fea_list(vid_i).name));
            feaMap(fea_list(vid_i).name) = tmpfea;
        end
    end
else
    fprintf('Load existing feaMap.\n');
    load('feaMap.mat')
end
%%Load all gesture labels
%gesMap = containers.Map;
%vid_list = dir([fea_path,'/*.mp4']);
%for vid_i = 1:length(vid_list)
%    name = vid_list(vid_i).name(1:end-4);
%    eval(sprintf('gesMap(name)=str2num(fea_dict.%s;',name));
fprintf('Finished loading feature.\n');

% Train gesture classifiers
gesture_score = cell(1,10);
gesture_acc = cell(1,10);
gesture_pred = cell(1,10);
for cv = 0:9
    gesture_score{cv+1} = cell(1, num_ges);
    gesture_acc{cv+1} = cell(1, num_ges);
    gesture_pred{cv+1} = cell(1, num_ges);
    for ges_i = 1:num_ges
        fprintf('CV: %d, ges_id: %d\n', cv, ges_i);
        trainfile = ['/media/bharat/HDD/zhe/Real-life_Deception_Detection_2016/Scripts_by_ID/trainVideo',num2str(cv),'.txt'];
        testfile = ['/media/bharat/HDD/zhe/Real-life_Deception_Detection_2016/Scripts_by_ID/testVideo',num2str(cv),'.txt'];
        fid = fopen(trainfile);
        C = textscan(fid, '%s');
        fclose(fid);
        num_v = length(C{1})/2;
        fea_mat = zeros(num_v, 24576*2);
        lab_vec = zeros(num_v, 1);
        for i = 1:num_v
            [pathstr,name,ext] = fileparts(C{1}{2*i-1});
            videoname = [name,ext];
            eval(sprintf('lab_vec(i)=str2num(fea_dict.%s(%d));',name,ges_i));
            fea = [];
            for j = 1: length(fea_types)
                fea_file = [videoname,'.',fea_types{j},'.fv.txt'];%fullfile(fea_path,[videoname,'.',fea_types{j},'.fv.txt']);
                fea = [fea, feaMap(fea_file)];
                %disp(fea_file)
                %fea = [fea, importdata(fea_file)];
            end
            fea_mat(i,:) = fea;
        end
        model = svmtrain(lab_vec, fea_mat, '-t 0 -q');
        %test phase
        fid = fopen(testfile);
        C = textscan(fid, '%s');
        fclose(fid);
        num_v = length(C{1})/2;
        fea_mat = zeros(num_v, 24576*2);
        lab_vec = zeros(num_v, 1);
        for i = 1:num_v
            [pathstr,name,ext] = fileparts(C{1}{2*i-1});
            videoname = [name,ext];
            eval(sprintf('lab_vec(i)=str2num(fea_dict.%s(%d));',name,ges_i));
            fea = [];
            for j = 1: length(fea_types)
                fea_file = [videoname,'.',fea_types{j},'.fv.txt']%fullfile(fea_path,[videoname,'.',fea_types{j},'.fv.txt']);
                %disp(fea_file)
                fea = [fea, feaMap(fea_file)];
                %fea = [fea, importdata(fea_file)];
            end
            fea_mat(i,:) = fea;
        end
        [pred, acc, prob] = svmpredict(lab_vec,fea_mat, model);
        pred
        gesture_score{cv+1}{ges_i} = prob;
        gesture_acc{cv+1}{ges_i} = acc;
        gesture_pred{cv+1}{ges_i} = pred;
        %[AUC, tt, tf, ft, ff] = auc_fun(lab_vec, prob, pred);
        %gesture_auc{cv+1}{ges_i} = AUC;
    end
    cv_gesture_score = gesture_score{cv+1};
    save(['gesture_score',num2str(cv),'.mat'],'cv_gesture_score');
end
save('total_gesture_score.mat', 'gesture_score');
save('total_gesture_pred.mat', 'gesture_pred');




