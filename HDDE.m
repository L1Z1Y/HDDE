clc;
clear;

format long;
format compact;

path = ".\results\cec2017\";
algName = "hdde";
runs = 51;

% record convergence data every 1/(converNum - 1)
converNum = 21;

funcNum = 30;

for dim = [10 30 50 100]
    maxNfes = 10000 * dim;

    rng('default');

    val2Reach = 10^(-8);
    lb = -100 * ones(1, dim);
    ub = 100 * ones(1, dim);
    lu = [lb; ub];
    fhd = @cec17_func;
    output = zeros(funcNum, 3);
    output(:, 1) = (1:funcNum)';

    % record the results of multiple experiments.
    fileName = path + algName + "\" + dim  + "D_output-" + algName + ".xlsx";
    % record statistical results of multiple experiments
    filenameMeanStd = path + algName + "\" + dim  + "D_output-" + algName + "-ms.xlsx";

    % if the folder does not exist, create the corresponding folder.
    [folderPath, ~, ~] = fileparts(fileName);
    if ~exist(folderPath, 'dir')
        mkdir(folderPath);
    end

    for funcNo = 1:funcNum
        % the optimum value for each function
        optimum = funcNo * 100.0;

        % record the best results
        outcome = [];

        convergence = zeros(runs, converNum);
        % record the average convergence curve for multiple experiments
        converFile = ".\results\cec2017\convergence\" + dim + "D\" + dim + "D_F" + funcNo + ".xlsx";

        fprintf('\n-------------------------------------------------------\n')
        fprintf('Function = %d, Dimension size = %d\n', funcNo, dim)

        parfor runId = 1 : runs
            conver = zeros(1, converNum);
            % dividing line parameter
            s = 0.7;
            % parameter settings for L-SHADE
            pBestRate = 0.11;
            arcRate = 2.6;
            memorySize = 5;
            m = 6;
            popSize = round(25 * log(dim) * sqrt(dim));
            pMin = 0.125;
            pMax = 0.25;

            maxPopSize = popSize;
            minPopSize = 4.0;

            %% initialize the main population
            popOld = repmat(lb, popSize, 1) + rand(popSize, dim) .* (repmat(ub - lb, popSize, 1));
            % the old population becomes the current population
            pop = popOld;
            count = zeros(popSize, 1);
            dis = (ub(1) - lb(1)) / 2 * sqrt(dim) * ones(popSize, 1);

            fitness = fhd(popOld', funcNo);
            fitness = fitness';
            nfes = popSize;

            [bsfFitVar, ii] = min(fitness);
            bsfSol = pop(ii, :);

            memorySf = 0.3 .* ones(memorySize, 1);
            memoryCr = 0.8 .* ones(memorySize, 1);
            memoryPos = 1;

            % the maximum size of the archive
            archive.NP = arcRate * popSize;
            % the solutions stored in te archive
            archive.pop = zeros(0, dim);
            % the function value of the archived solutions
            archive.funvalues = zeros(0, 1);

            cur = 1;
            conver(cur) = bsfFitVar - optimum;
            if conver(cur) < val2Reach
                conver(cur) = 0;
            end
            %% main loop
            while nfes < maxNfes
                % the old population becomes the current population
                pop = popOld;
                [tempFit, sortedIndex] = sort(fitness, 'ascend');

                memRandIndex = ceil(memorySize * rand(popSize, 1));
                muSf = memorySf(memRandIndex);
                muCr = memoryCr(memRandIndex);

                % Innovative points of iLSHADE
                muCr(memRandIndex == memorySize) = 0.9;
                muSf(memRandIndex == memorySize) = 0.9;
                pBestRate = (pMax-pMin) * (nfes/maxNfes) + pMin;

                % for generating crossover rate
                cr = normrnd(muCr, 0.1);
                termPos = find(muCr == -1);
                cr(termPos) = 0;
                cr = min(cr, 1);
                % Innovative points of iLSHADE
                if nfes < 0.25 * maxNfes
                    cr = max(cr, 0.7);
                elseif nfes < 0.5 * maxNfes
                    cr = max(cr, 0.6);
                else
                    cr = max(cr, 0);
                end

                %% for generating scaling factor
                sf = muSf + 0.1 * tan(pi * (rand(popSize, 1) - 0.5));
                pos = find(sf <= 0);

                while ~ isempty(pos)
                    sf(pos) = muSf(pos) + 0.1 * tan(pi * (rand(length(pos), 1) - 0.5));
                    pos = find(sf <= 0);
                end

                stage = 0.6;
                sfCurMax = 0.6;
                if nfes / maxNfes >= stage
                    x = (nfes / maxNfes - stage) / (1 - stage);
                    sfCurMax = 0.6 + (1 - (x - 1) ^ 2) ^ (1 / 2) * (1 - 0.6);
                end

                sf = min(sf, sfCurMax);

                % jSO
                if nfes < 0.2 * maxNfes
                    fw = 0.7 * sf;
                elseif nfes < 0.4 * maxNfes
                    fw = 0.8 * sf;
                else
                    fw = 1.2 * sf;
                end

                r0 = 1 : popSize;
                if nfes / maxNfes >= s
                    popAll = [pop; archive.pop];
                    [r1, r2] = gnR1R2(popSize, size(popAll, 1), r0);
                else
                    popAll = pop;
                    [r1, r2] = gnBetterR1R2(popSize, 1:popSize, fitness);
                end
                % choose at least two best solutions
                pNP = max(round(pBestRate * popSize), 2);
                % select from [1, 2, 3, ..., pNP] Find random x_pbest for each individual
                randindex = ceil(rand(1, popSize) .* pNP);
                % to avoid the problem that rand = 0 and thus ceil(rand) = 0
                randindex = max(1, randindex);
                % randomly choose one of the top 100p% solutions
                pbest = pop(sortedIndex(randindex), :);

                vi = pop + sf(:, ones(1, dim)) .* (pop(r1, :) - popAll(r2, :)) + fw(:, ones(1, dim)) .* (pbest - pop);
                vi = boundConstraint(vi, pop, lu);
                % mask is used to indicate which elements of ui comes from the parent
                mask = rand(popSize, dim) > cr(:, ones(1, dim));
                % choose one position where the element of ui doesn't come from the parent
                rows = (1 : popSize)'; cols = floor(rand(popSize, 1) * dim) + 1;
                jrand = sub2ind([popSize dim], rows, cols); mask(jrand) = false;
                ui = vi; ui(mask) = pop(mask);

                childFitness = fhd(ui', funcNo);
                childFitness = childFitness';

                %%%%%%%%%%%%%%%%%%%%%%%%
                for i = 1 : popSize
                    nfes = nfes + 1;

                    if childFitness(i) < bsfFitVar
                        bsfFitVar = childFitness(i);
                        bsfSol = ui(i, :);
                    end

                    if floor(nfes / (maxNfes / (converNum - 1))) >= cur
                        cur = cur + 1;
                        conver(cur) = bsfFitVar - optimum;
                        if conver(cur) < val2Reach
                            conver(cur) = 0;
                        end
                    end

                    if nfes > maxNfes; break; end
                end
                %%%%%%%%%%%%%%%%%%%%%%%% for out

                dif = abs(fitness - childFitness);

                %% I == 0: the parent is better; I == 1: the offspring is better
                I = (fitness > childFitness);
                goodCR = cr(I == 1);
                goodF = sf(I == 1);
                difVal = dif(I == 1);
                distI = sqrt(sum((popOld(I == 1,:) - ui(I == 1,:)).^2,2));
                dis(I == 1) = distI;
                w = distI / sum(distI);
                archive = updateArchive(archive, popOld(I == 1, :), fitness(I == 1));
                %% I == 1: the parent is better; I == 2: the offspring is better
                [fitness, I] = min([fitness, childFitness], [], 2);
                
                % ESDE
                for i = 1 : size(I,1)
                    if I(i,:) == 1
                        count(i) = count(i) + 1;
                        d = sqrt(sum((popOld(i,:) - ui(i,:)).^2,2));
                        dc = d / dis(i);
                        [alpha, beta, gamma, dRate] = acParamCal(dim, nfes, maxNfes, dc, count(i))
                        acRate = dRate * alpha * (1 / (1 + exp(beta - count(i))) * (1 / (1 + exp(nfes - gamma * maxNfes))));
                        if rand < acRate
                            I(i,:) = 2;
                            fitness(i, :) = childFitness(i, :);
                        end
                    else
                        count(i) = 0;
                    end
                end

                popOld = pop;
                popOld(I == 2, :) = ui(I == 2, :);

                numSuccessParams = numel(goodCR);

                if numSuccessParams > 0
                    sumDif = sum(difVal);
                    difVal = difVal / sumDif;

                    % for updating the memory of scaling factor
                    memorySf(memoryPos) = ((w' * (goodF .^ 2)) / (w' * goodF) + memorySf(memoryPos)) / 2;

                    % for updating the memory of crossover rate
                    if max(goodCR) == 0 || memoryCr(memoryPos)  == -1
                        memoryCr(memoryPos)  = -1;
                    else
                        memoryCr(memoryPos) = ((w' * (goodCR .^ 2)) / (w' * goodCR) + memoryCr(memoryPos)) / 2;
                    end

                    memoryPos = memoryPos + 1;
                    if memoryPos > memorySize;  memoryPos = 1; end
                end

                %% resizing the population size
                planPopSize = round((((minPopSize - maxPopSize) / maxNfes) * nfes) + maxPopSize);

                if popSize > planPopSize
                    reductionIndNum = popSize - planPopSize;
                    if popSize - reductionIndNum <  minPopSize; reductionIndNum = popSize - minPopSize;end

                    popSize = popSize - reductionIndNum;
                    for r = 1 : reductionIndNum
                        [valBest, indBest] = sort(fitness, 'ascend');
                        worstInd = indBest(end);
                        popOld(worstInd, :) = [];
                        pop(worstInd, :) = [];
                        fitness(worstInd, :) = [];
                        count(worstInd, :) = [];
                    end

                    archive.NP = round(arcRate * popSize);

                    if size(archive.pop, 1) > archive.NP
                        rndPos = randperm(size(archive.pop, 1));
                        rndPos = rndPos(1 : archive.NP);
                        archive.pop = archive.pop(rndPos, :);
                    end
                end

            end

            bsfErrorVal = bsfFitVar - optimum;
            if bsfErrorVal < val2Reach
                bsfErrorVal = 0;
            end

            convergence(runId, :) = conver;

            if conver(converNum) ~= bsfErrorVal
                fprintf("out' best is %f and conver's best is %f\n", bsfErrorVal, conver(converNum))
                conver(converNum) = bsfErrorVal;
            end

            fprintf('%d th run, best-so-far error value = %1.8e\n', runId, bsfErrorVal)
            outcome = [outcome bsfErrorVal];
        end % end 1 run

        fprintf('\nmean error value = %1.8e, std = %1.8e\n', mean(outcome), std(outcome))
        range = sprintf('A%d', funcNo);
        % write the 
        writematrix(outcome, fileName, 'Range', range);
        output(funcNo,2:3) = [mean(outcome), std(outcome)];

        converTable = array2table(mean(convergence), 'RowNames', algName);
        % write the average convergence curve data from multiple experiments to the hard disk
        % writetable(converTable, converFile, 'WriteMode', 'append', 'WriteRowNames', true, "WriteVariableNames", false);
    end % end 1 function run
    columnNames = {'func', 'mean', 'std'};
    dataTable = array2table(output, 'VariableNames', columnNames);
    writetable(dataTable, filenameMeanStd);
end


function [r1, r2] = gnBetterR1R2(NP1, r0, fitness)
NP0 = length(r0);

n1 = ceil(NP1 / 2);
n2 = NP1 - n1;

[~, fi] = sort(fitness);
fi1 = fi(1:n1);
fi2 = fi(n1+1:NP1);

r1 = fi1(ceil(rand(1, NP0) * n1));
r1 = r1';
for i = 1 : 9999
    pos = (r1 == r0);
    if sum(pos) == 0
        break;
    else % regenerate r1 if it is equal to r0
        r1(pos) = fi1(ceil(rand(1, sum(pos)) * n1));
    end
    if i > 1000 % this has never happened so far
        fprintf("NP1 = [%d]", NP1);
        error('Can not genrate r1 in 1000 iterations');
    end
end

r2 = fi2(ceil(rand(1, NP0) * n2));
r2 = r2';
for i = 1 : 9999
    pos = ((r2 == r1) | (r2 == r0));
    if sum(pos)==0
        break;
    else % regenerate r2 if it is equal to r0 or r1
        r2(pos) = fi2(ceil(rand(1, sum(pos)) * n2));
    end
    if i > 1000 % this has never happened so far
        error('Can not genrate r2 in 1000 iterations');
    end
end
end


function [alpha, beta, gamma, dRate] = acParamCal(dim, nfes, maxNfes, dc, count)
alpha = 0.5;
gamma = 0.4;
betaMin = 24 + (dim == 10) * 24;
betaMax = 208 - (dim == 100) * 48;
if dim == 50 || dim == 100
    gamma = 0.5;
end
beta = betaMin + (nfes / maxNfes >= 0.5) * ((nfes - 0.5 * maxNfes) / (0.5 * maxNfes)) * (betaMax - betaMin);
dRate = min(dc, 2 + 10 * (count >= beta));
end

