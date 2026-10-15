%% Between Subjects Bootstrapped Bayesian model testing
% ==================================== Developed by AK, MRB, Mar 2024
% ==================================== Last updated for OSF: July 2025, MRB
%% load data and age vector
load('data4BayesModels.mat')
% data4Bayes dimensions: 129 x 51 x 6
    % 1. electrodes
    % 2. infant participants, stacked such that 6mo; 9mo; 12mo
    % 3. conditions: Individual trained, Indiv Untrained, Category trained, Cat untrained, No Label Trained, No Label Untrained
%% ====================== Linear Model of Trained - Untrained Difference ===============================
% load data in correct format for Trained - Untrained Difference
    % i.e., px in the 1st dim.
    c = 1;
    for label = 1:2:5
        tmpData = data4Bayes(:,:,label) - data4Bayes(:,:,label+1); % label: trained; label+1: untrained
        data2glm(:,:,c) = transpose(tmpData);
        c = c+1;
    end
    % NOTE data2glm: infant (51) x elec x label (data are the trained - untrained difference)
% model parameters
    weights = [1 2 3]; % the models, weights across age
    ndraws = 5000; % number of draws/iterations for bootstrapping
% age group matrix creation 
    % index = in stacked data
    ages = [ones(15,1).*6; ones(17,1).*9;ones(19,1).*12];
    index6mo = find(ages==6); size6mo = size(index6mo,1); grp6 = data2glm(index6mo,:,:); 
    index9mo = find(ages==9); size9mo = size(index9mo,1); grp9 = data2glm(index9mo,:,:); 
    index12mo = find(ages==12); size12mo = size(index12mo,1); grp12 = data2glm(index12mo,:,:);    
tic % gentlefolks, start the clock
% make bootstrap distributions of inner products
    for draw = 1:ndraws
        if draw/100 == round(draw./100), fprintf(['draw: ' num2str(draw) ' of ' num2str(ndraws) '\n']), end % progress indicator
        for label = 1:size(data2glm,3)
            for model = 1:size(weights,1)
            % get group means, px drawn with replacement
                means4bootstrap = [mean(grp6(randi(size6mo,1,size6mo),:,label))' mean(grp9(randi(size9mo,1,size9mo),:,label))' mean(grp12(randi(size12mo,1,size12mo),:,label))'];
            % for tested models
                innerprod_tested(:,draw,label,model) = means4bootstrap * weights(model,:)';
            % for null model (bootstrapped people, shuffled weights):
                innerprod_null(:,draw,label,model) =  means4bootstrap * weights(model,randperm(size(weights,2)))';
            end % model
        end % condition
    end % draws
    fprintf('Done with Bootstrapping!\n')
% get bayes factors for model vs null distributions
    for model = 1:size(weights,1)
        for label = 1:size(data2glm,3)
            for elec = 1:size(data2glm,2)
                BF10_trainedUntrained(elec,label,model) = bootstrap2BF_z(innerprod_tested(elec,:,label,model),innerprod_null(elec,:,label,model), 0);
                BF01_trainedUntrained(elec,label,model) = bootstrap2BF_z(innerprod_null(elec,:,label,model),innerprod_tested(elec,:,label,model), 0);
            end % elec
        end % label
    end % model
    fprintf('Done with BF calculation!\n')
%% ====================== Competition and Amplification Models ===============================
% load data in correct format => competition and amplification models
    for label = 1:6
        tmpData = data4Bayes(:,:,label); 
        data2glm(:,:,label) = transpose(tmpData);
    end
    data2glm2(:,:,1,1) = data2glm(:,:,1); data2glm2(:,:,2,1) = data2glm(:,:,2); % indiviudal
    data2glm2(:,:,1,2) = data2glm(:,:,3); data2glm2(:,:,2,2) = data2glm(:,:,4); % category
    data2glm2(:,:,1,3) = data2glm(:,:,5); data2glm2(:,:,2,3) = data2glm(:,:,6); % no label
% ==> data2glm2: 51 x 129 x 2 (trained/untrained) x 3 (label)
% model parameters
    weights = [3 3 4 2 5 1; 3 3 4 3 5 3]; % competition vs amplitifcation; weights across Tr/unt & age
    ndraws = 5000; % number of draws/iterations for bootstrapping
% group matrix creation 
    index6mo = find(ages==6); size6mo = size(index6mo,1); grp6 = data2glm2(index6mo,:,:,:); 
    index9mo = find(ages==9); size9mo = size(index9mo,1); grp9 = data2glm2(index9mo,:,:,:); 
    index12mo = find(ages==12); size12mo = size(index12mo,1); grp12 = data2glm2(index12mo,:,:,:);
tic % gentlefolks, start the clock
% make bootstrap distributions of inner products
    for draw = 1:ndraws
        if draw/100 == round(draw./100), fprintf(['draw: ' num2str(draw) ' of ' num2str(ndraws) '\n']), end % progress indicator
        for label = 1:size(data2glm2,4)
            for model = 1:size(weights,1)
            % get group means
                means4bootstrap = [squeeze(mean(grp6(randi(size6mo,1,size6mo),:,:,label))) ...
                    squeeze(mean(grp9(randi(size9mo,1,size9mo),:,:,label))) ...
                    squeeze(mean(grp12(randi(size12mo,1,size12mo),:,:,label)))];
             % for tested models
                innerprod_tested(:,draw,label,model) = means4bootstrap * weights(model,:)';
             % for null models
                innerprod_null(:,draw,label,model) =  means4bootstrap * weights(model,randperm(size(weights,2)))';
            end % model
        end % label
    end % draws
    fprintf('Done with Bootstrapping!\n')
% get bayes factors for model vs null distributions
    for model = 1:size(weights,1)
        for label = 1:size(data2glm2,4)
            for elec = 1:size(data2glm2,2)
                BF10_compAmp(elec,label,model) = bootstrap2BF_z(innerprod_tested(elec,:,label,model),innerprod_null(elec,:,label,model), 0);
                BF01_compAmp(elec,label,model) = bootstrap2BF_z(innerprod_null(elec,:,label,model),innerprod_tested(elec,:,label,model), 0);
            end % elec
        end % label
    end % model
fprintf('Done with BF calculation!\n')
%% ==================== Plotting topographic plots of elecs with BF10 >= 3 ====================
% load correct elec montage, get colors for BF >= 3 elec markers
    load locsEEGLAB129HCL
    colors4topo = turbo(12); colors4topo = colors4topo([8 9 11],:); 
% define BF data and titles according to the model (uncomment active model)
    % Linear Model
        dataBF = BF10_trainedUntrained; title4plot = 'Linear Model'; model = 1;
        bfValSplits = [3 4 5]; % values that split elecs by color
    % Competition and Amplification Models
        % dataBF = BF10_compAmp(:,:,1); title4plot = 'Competition Model'; model = 2;
        % dataBF = BF10_compAmp(:,:,2); title4plot = 'Amplification Model'; model = 3;
        % bfValSplits = [3 5 7];
% label names
    lNames = {'Individual','Category','No Label'};
% identify which elecs, per label, are >= 3, create color vector for topoplot
    sigElec_Idx = [];
    for label = 1:3
        tempElecs = find(dataBF(:,label) >= 3);
        sigElec_Idx{label} = tempElecs;
        colorVec_tmp =[];
        BFvalues = dataBF(tempElecs,label);
        for x = 1:size(BFvalues,1)
            if BFvalues(x) >= bfValSplits(1) && BFvalues(x) < bfValSplits(2), colorVec_tmp(x,:) = colors4topo(1,:);
            elseif BFvalues(x) >= bfValSplits(2) && BFvalues(x) < bfValSplits(3), colorVec_tmp(x,:) = colors4topo(2,:);
            elseif BFvalues(x) >= bfValSplits(3), colorVec_tmp(x,:) = colors4topo(3,:);
            end
        end
        colorVec{label} = colorVec_tmp;
    end
% plot "blank" topoplots (EEGLAB) with only the suprathres elec identified, multicolored
    for label = 1:3
        figure(model)
        subplot(1,3,label)
        topoplotMB(zeros(129,1),locsEEGLAB129HCL,'style','straight','electrodes','on', 'maplimits',[0 7],...
            'emarker2',{sigElec_Idx{label},'o',colorVec{label},19,1},'hcolor','k','emsize',25,'ecolor','k','colormap',white);
        title([title4plot 'Suprathreshold Electrodes (BF >= 3)' lNames(label)],'FontSize',18)
    end
%% ta da !!! :)