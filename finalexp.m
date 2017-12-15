function [AUC_FV,AUC_ges,AUC_GT,AUC_trans]=finalexp(method)
%This script run different methods on different sets of features to fill the final table
% method may be 'NN'(network),'tree','randforest','bayes','log','boost', 'linearsvm','kernelsvm'
%csv_path = '../Annotation/saved_anno.mat';
%load(csv_path);

disp(['Classifier is ', method])
% read CV
fea_path = '../dataset_trial/FVs';
fea_types = {'mbhx','mbhy'};

% Load all FV features
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
    save('feaMap.mat','feaMap');
else
    load('feaMap.mat');
end

prob_lf = cell(1, 10);
prob_FV = cell(1, 10);  %Fisher Vector
prob_ges = cell(1,10);  %pred gesture
prob_GT = cell(1, 10);  %GT gesture
prob_MFCC = cell(1,10); %MFCC feature
prob_trans = cell(1,10);    %Transcript feature
lab_vec = cell(1, 10);  %label vector
AUC_gt = zeros(1,10);

%fine grid search on gesture features on different weights (0 0.5 1)
for cv = 0:9
    disp(['CROSS VALIDATION:', num2str(cv)])
    if ~exist(sprintf(['grid_search/final%d_',method,'.mat'],cv+1),'file') 
        disp('IDTFV');
        tmpprob_FV = test_dtfv_CV(cv, feaMap, method);  %IDT+FV
        disp('Pred');
        [tmplab,tmpprob, w] = test_gestureSearch(cv, method); %Pred Gesture
        disp('GT')
        tmpprob_GT = test_GTgestureSearch(cv, method);  %GT Gesture, currently use 5 GT
        disp('trans');
        tmpprob_trans = test_Trans(cv, method);
        save(sprintf(['grid_search/final%d_',method,'.mat'],cv+1), 'tmpprob_FV','tmplab','tmpprob','tmpprob_GT','tmpprob_trans');
        fprintf('Save search results.\n');
    else
        disp(['Load from ',sprintf(['grid_search/final%d_',method,'.mat'],cv+1)]);
        load(sprintf(['grid_search/final%d_',method,'.mat'],cv+1));
    end
%     [tmplab,tmpprob, w] = test_gestureSearch(cv, method); %Pred Gesture
    lab_vec{cv+1} = tmplab;
    prob_ges{cv+1} = tmpprob;
    prob_FV{cv+1} = tmpprob_FV;
    prob_GT{cv+1} = tmpprob_GT;
    prob_trans{cv+1} = tmpprob_trans;
    if ~exist(sprintf(['grid_search/MFCC%d_',method,'.mat'],cv+1),'file')
        disp('MFCC');
        tmpprob_MFCC = test_MFCC(cv, method);
        save(sprintf(['grid_search/MFCC%d_',method,'.mat'],cv+1), 'tmpprob_MFCC');
        fprintf('Save MFCC.\n');
    else
        disp(['Load from ',sprintf(['grid_search/MFCC%d_',method,'.mat'],cv+1)]);
        load(sprintf(['grid_search/MFCC%d_',method,'.mat'],cv+1));   
    end
    prob_MFCC{cv+1} = tmpprob_MFCC;   
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
% Compute individual feature AUC
for cv = 0:9
    validID = ~isnan(prob_ges{cv+1}{length(prob_ges{1})});
    % IDTFV only
    [~,~,~,AUC_FV(cv+1)] = perfcurve(lab_vec{cv+1}(validID),prob_FV{cv+1}(validID),0);
    % Gesture only
    for comb_i = 1: length(prob_ges{1})
        [~,~,~,AUC_ges(cv+1, comb_i)] = perfcurve(lab_vec{cv+1}(validID),prob_ges{cv+1}{comb_i}(validID),0);
        [~,~,~,AUC_GT(cv+1, comb_i)] = perfcurve(lab_vec{cv+1}(validID),prob_GT{cv+1}{comb_i}(validID),0);
    end
    % Trans only
    [~,~,~,AUC_trans(cv+1)] = perfcurve(lab_vec{cv+1}(validID),prob_trans{cv+1}(validID),0);
    % MFCC only
    [~,~,~,AUC_MFCC(cv+1)] = perfcurve(lab_vec{cv+1}(validID),prob_MFCC{cv+1}(validID),0);

end

fprintf('IDTFV only: %f.\n', mean(AUC_FV));
fprintf('Pred Gesture only: %f.\n', max(mean(AUC_ges,1)));
fprintf('GT Gesture only: %f.\n', max(mean(AUC_GT,1)));
fprintf('Trans only: %f.\n', mean(AUC_trans));
fprintf('MFCC only: %f.\n', mean(AUC_MFCC));

tmpAUC = mean(AUC_ges,1); comb_maxi = find(tmpAUC==max(tmpAUC)); comb_maxi = comb_maxi(1); %for gesture and GT
tmpAUC = mean(AUC_GT,1); comb_maxi_GT = find(tmpAUC==max(tmpAUC)); comb_maxi_GT = comb_maxi_GT(1);

% Do a grid search to find best weights for ges_id
combs = unique(combnk([0 0 0 0 0 .5 .5 .5 .5 .5 1 1 1 1 1], 5), 'rows');
comb_i = 1;
for i = 1:size(combs,1)
    perW = unique(perms(combs(i,:)), 'rows');
    for j = 1:size(perW, 1)
        %tmpw = repmat(perW(j,:), num_C, 1);
        w{comb_i} = perW(j,:);
        comb_i = comb_i +1;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Find best ratio for idt and Gesture
disp('---------------------------------------------------------------------------------')
disp('IDT+ Gesture...............')
for comb_i = 1: length(prob_ges{1}) 
    for r  = 1:11
        ratio = -0.1+ 0.1*r;
        AUC = zeros(1, 10);
        AUC_GT = zeros(1, 10);
        for cv = 0:9
            % pred late fusion
            prob_lf= ratio*prob_FV{cv+1}(~isnan(prob_ges{cv+1}{comb_i})) +(1-ratio)* prob_ges{cv+1}{comb_i}(~isnan(prob_ges{cv+1}{comb_i})); 
            % GT late fusion
            if ~isinf(prob_GT{cv+1}{comb_i}(~isnan(prob_ges{cv+1}{comb_i})))
            %prob_FV{cv+1}(~isnan(prob_ges{cv+1}{comb_i}))
            %prob_GT{cv+1}{comb_i}(~isnan(prob_ges{cv+1}{comb_i}))
                prob_lf_GT = ratio*prob_FV{cv+1}(~isnan(prob_ges{cv+1}{comb_i})) +(1-ratio)* prob_GT{cv+1}{comb_i}(~isnan(prob_ges{cv+1}{comb_i})); 
            else
                prob_lf_GT = ratio*prob_FV{cv+1}(~isnan(prob_ges{cv+1}{comb_i}));
            end
            [~,~,~,AUC(cv+1)] = perfcurve(lab_vec{cv+1}(~isnan(prob_ges{cv+1}{comb_i})),prob_lf,0);
            [~,~,~,AUC_GT(cv+1)] = perfcurve(lab_vec{cv+1}(~isnan(prob_ges{cv+1}{comb_i})),prob_lf_GT,0);
        end
        AUC_r(comb_i, r) = mean(AUC);
        AUC_r_GT(comb_i,r) = mean(AUC_GT);
    end
end
% find best ratio and gesture weights value, only select one if more than one are applicable
[comb_s, r_s] = find(AUC_r == max(AUC_r(:)));
[comb_s_GT, r_s_GT] = find(AUC_r_GT == max(AUC_r_GT(:)));
combs = unique(combnk([0 0 0 0 0 .5 .5 .5 .5 .5 1 1 1 1 1], 5), 'rows');
comb_i = 1;
for i = 1:size(combs,1)
    perW = unique(perms(combs(i,:)), 'rows');
    for j = 1:size(perW, 1)
        %tmpw = repmat(perW(j,:), num_C, 1);
        w{comb_i} = perW(j,:);
        comb_i = comb_i +1;
    end
end
comb_s = comb_s(1);
r_s = r_s(1);
comb_s_GT = comb_s_GT(1);
r_s_GT = r_s_GT(1);
fprintf('IDTFV+Pred: %f.\n', max(AUC_r(:)));

ratio1 = -0.1+0.1*r_s; % ratio for idt and gesture
ratio1_GT = -0.1+0.1*r_s_GT; % ratio for idt and GT gesture

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Also consider MFCC feature
disp('---------------------------------------------------------------------------------')
disp('With MFCC Feature......................')
for r = 1:11
    ratio2 = -0.1+0.1*r;
    AUC = zeros(1, 10);
    AUC_GT = zeros(1,10);
    for cv  = 0:9
        % pred+idt probs and GT+idt probs
        prob_lf0 = ratio1*prob_FV{cv+1}(~isnan(prob_ges{cv+1}{comb_s})) +(1-ratio1)* prob_ges{cv+1}{comb_s}(~isnan(prob_ges{cv+1}{comb_s}));
        prob_lf0_GT = ratio1_GT*prob_FV{cv+1}(~isnan(prob_ges{cv+1}{comb_s_GT})) +(1-ratio1_GT)* prob_GT{cv+1}{comb_s_GT}(~isnan(prob_ges{cv+1}{comb_s_GT}));
        % compute fusion
        prob_lf= ratio2*prob_lf0 + (1-ratio2)*prob_MFCC{cv+1}(~isnan(prob_ges{cv+1}{comb_s}));
        [~,~,~,AUC(cv+1)] = perfcurve(lab_vec{cv+1}(~isnan(prob_ges{cv+1}{comb_s})),prob_lf,0);
        prob_lf_GT= ratio2*prob_lf0_GT + (1-ratio2)*prob_MFCC{cv+1}(~isnan(prob_ges{cv+1}{comb_s_GT}));
        [~,~,~,AUC_GT(cv+1)] = perfcurve(lab_vec{cv+1}(~isnan(prob_ges{cv+1}{comb_s_GT})),prob_lf_GT,0);
    end
    AUC_lf(r) = mean(AUC(~isnan(AUC)));
    AUC_lf_GT(r) = mean(AUC_GT(~isnan(AUC_GT)));
end

fprintf('IDTFV+Pred+MFCC: %f.\n', max(AUC_lf(:)));
fprintf('IDTFV+GT+MFCC: %f.\n', max(AUC_lf_GT(:)));

r_s = find(AUC_lf == max(AUC_lf));
r_s_GT = find(AUC_lf_GT == max(AUC_lf_GT));
r_s_GT = r_s_GT(1);
r_s = r_s(1);

ratio_M = -0.1+0.1*r_s; % ratio for idt+ges and MFCC
ratio_M_GT = -0.1+0.1*r_s_GT; % ratio for idt+GT and MFCC


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Also consider Trans feature
disp('---------------------------------------------------------------------------------')
disp('With Trans Feature......................')
for r = 1:11
    ratio2 = -0.1+0.1*r;
    AUC = zeros(1, 10);
    AUC_GT = zeros(1,10);
    for cv  = 0:9
        % pred+idt probs and GT+idt probs
        prob_lf0 = ratio1*prob_FV{cv+1}(~isnan(prob_ges{cv+1}{comb_s})) +(1-ratio1)* prob_ges{cv+1}{comb_s}(~isnan(prob_ges{cv+1}{comb_s}));
        prob_lf0_GT = ratio1_GT*prob_FV{cv+1}(~isnan(prob_ges{cv+1}{comb_s_GT})) +(1-ratio1_GT)* prob_GT{cv+1}{comb_s_GT}(~isnan(prob_ges{cv+1}{comb_s_GT}));
        % compute fusion
        prob_lf= ratio2*prob_lf0 + (1-ratio2)*prob_trans{cv+1}(~isnan(prob_ges{cv+1}{comb_s}));
        [~,~,~,AUC(cv+1)] = perfcurve(lab_vec{cv+1}(~isnan(prob_ges{cv+1}{comb_s})),prob_lf,0);
        prob_lf_GT= ratio2*prob_lf0_GT + (1-ratio2)*prob_trans{cv+1}(~isnan(prob_ges{cv+1}{comb_s_GT}));
        [~,~,~,AUC_GT(cv+1)] = perfcurve(lab_vec{cv+1}(~isnan(prob_ges{cv+1}{comb_s_GT})),prob_lf_GT,0);
    end
    AUC_lf(r) = mean(AUC(~isnan(AUC)));
    AUC_lf_GT(r) = mean(AUC_GT(~isnan(AUC_GT)));
end

fprintf('IDTFV+Pred+trans: %f.\n', max(AUC_lf(:)));
fprintf('IDTFV+GT+trans: %f.\n', max(AUC_lf_GT(:)));

r_s = find(AUC_lf == max(AUC_lf));
r_s_GT = find(AUC_lf_GT == max(AUC_lf_GT));
r_s_GT = r_s_GT(1);
r_s = r_s(1);

ratio_T = -0.1+0.1*r_s; % ratio for idt+ges and MFCC
ratio_T_GT = -0.1+0.1*r_s_GT; % ratio for idt+GT and MFCC

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% All features
disp('---------------------------------------------------------------------------------')
disp('With All Features...................')
for r = 1:11
    ratio2 = -0.1+0.1*r;
    AUC = zeros(1, 10);
    AUC_GT = zeros(1,10);
    for cv  = 0:9
        % pred gesture with 
        prob_lf0 = ratio1*prob_FV{cv+1}(~isnan(prob_ges{cv+1}{comb_s})) +(1-ratio1)* prob_ges{cv+1}{comb_s}(~isnan(prob_ges{cv+1}{comb_s}));
        % GT gesture with trans
        prob_lf0_GT = ratio1_GT*prob_FV{cv+1}(~isnan(prob_ges{cv+1}{comb_s_GT})) +(1-ratio1_GT)* prob_GT{cv+1}{comb_s_GT}(~isnan(prob_ges{cv+1}{comb_s_GT}));
        % compute fusion with MFCC
        prob_M0 = ratio_M*prob_lf0 + (1-ratio_M)*prob_MFCC{cv+1}(~isnan(prob_ges{cv+1}{comb_s}));
        prob_M0_GT = ratio_M_GT*prob_lf0_GT + (1-ratio_M_GT)*prob_MFCC{cv+1}(~isnan(prob_ges{cv+1}{comb_s_GT}));
        % compute fusion with Trans
        prob_T= ratio2*prob_M0 + (1-ratio2)*prob_trans{cv+1}(~isnan(prob_ges{cv+1}{comb_s}));
        [~,~,~,AUC(cv+1)] = perfcurve(lab_vec{cv+1}(~isnan(prob_ges{cv+1}{comb_s})),prob_T,0);
        prob_T_GT= ratio2*prob_M0_GT + (1-ratio2)*prob_trans{cv+1}(~isnan(prob_ges{cv+1}{comb_s_GT}));
        [~,~,~,AUC_GT(cv+1)] = perfcurve(lab_vec{cv+1}(~isnan(prob_ges{cv+1}{comb_s_GT})),prob_T_GT,0);
    end
    AUC_final(r) = mean(AUC(~isnan(AUC)));
    AUC_final_GT(r) = mean(AUC_GT(~isnan(AUC_GT)));
end

fprintf('IDTFV+Pred+MFCC+Trans: %f.\n', max(AUC_final(:)));
fprintf('IDTFV+GT+MFCC+Trans: %f.\n', max(AUC_final_GT(:)));

r_f = find(AUC_final == max(AUC_final));
r_f_GT = find(AUC_final_GT == max(AUC_final_GT));
r_f_GT = r_f_GT(1);
r_f = r_f(1);

ratio_All = -0.1+0.1*r_f; % ratio for idt+ges and MFCC
ratio_All_GT = -0.1+0.1*r_f_GT; % ratio for idt+GT and MFCC
