% This script test under 10-fold Cross Validation, divided by ID, how dtfv works
function [AUC, tt, tf, ft, ff] = auc_fun(label, score, pred)
lie_id = find(score<0);
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
[~,~,~,AUC] = perfcurve(label,isign*score,0);
tt = sum(pred(label==1)==1);
tf = sum(pred(label==1)==0);
ft = sum(pred(label==0)==1);
ff = sum(pred(label==0)==0);

%fprintf('Partation:%d\tacc is %f, auc is %f, tt:%d, tf:%d, ft:%d, ff:%d\n',part, acc(1),AUC,tt,tf,ft,ff);
end
