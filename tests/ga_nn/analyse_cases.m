function [res, graphs, scores] = analyse_cases(root_dir, exp_templ, exp_num)
% cd(root_dir);

disp('==================================================================================================');
exps = dir(strcat(root_dir, '/', exp_templ, '*'));
cnum = length(exps);
res = [];
graphs = cell(1, exp_num);
figure;
hold on
for i=1:exp_num
    a = cell(1, cnum);
    %read data
    mean_hits = 0;
    mean_dev = 0;
    mean_best = [];
    min_rnum = 0;
    mean_improve = [];
    mean_best_val = 0;
    m_epoch = 0;
    m_time = 0;
    for j=1:cnum
        fname = strcat(root_dir, '/', sprintf('%s%d', exp_templ, j), '/', sprintf('_%d_ga_stat.txt',  i));
        a{j} = dlmread(fname);
		% check for octave
        rnum = size(a{j});
		if rnum(2) > 8
			a{j} = a{j}(1:end, end-7:end);
		end
        rnum = rnum(1);
        mean_hits = mean_hits + max(a{j}(1:end, 7));
        %tmp = (a{j}(1:end, 3) - a{j}(1:end, 6)) ./ a{j}(1:end, 3);
        %mean_dev = mean_dev + sqrt(sum(tmp.*tmp) / rnum);
        mean_dev = mean(abs(a{j}(1:end, 3) - a{j}(1:end, 6)));
        if j == 1 || rnum < min_rnum
            min_rnum = rnum;
        end
        %calc mean improvement
        ind = find(a{j}(1:end, 7) ~= 0);
        mean_improve = [mean_improve; a{j}(ind, 3) ./ a{j}(ind - 1, 3)];
        mean_best_val = mean_best_val + min(a{j}(1:end, 3));
        m_epoch = m_epoch + rnum;
        m_time = m_time + a{j}(end, end);
    end
    mean_hits = mean_hits / cnum;
    mean_dev = mean_dev / cnum;
    mean_best_val = mean_best_val / cnum;
    m_epoch = m_epoch / cnum;
    m_time = m_time / cnum;
    %mean_improve = mean_improve / cnum;
    
    %calc mean graph of best fitness
    mean_ff = zeros(min_rnum, 1);
    for j = 1:cnum
        mean_ff = mean_ff + interp1((1:length(a{j}(1:end, 3)))', a{j}(1:end, 3), (1:min_rnum)');
    end
    mean_ff = mean_ff ./ cnum;
    graphs{i} = mean_ff;

    %calc % of NN hits
    m_phits = mean_hits * 100 / m_epoch;
    
    %print results
    if i > 1
        disp('-------------------');
    end
    fprintf('Experiment %d results:\n', i);
    fprintf('Mean best ff achieved: %g\n', mean_best_val);
    fprintf('Mean epochs: %g\n', m_epoch);
    %expand time
    sec = m_time; mn = 0; h = 0;
    if sec > 60
        mn = floor(sec / 60);
        sec = sec - 60 * mn;
    end
    if mn > 60
        h = floor(mn / 60);
        mn = mn - 60 * h;
    end
    fprintf('Mean optimization time: %g sec (%g h %g min %g sec)\n', m_time, h, mn, sec);
    fprintf('Mean NN hits: %g\n', mean_hits);
    fprintf('Mean %% of NN hits: %g%%\n', m_phits);
    fprintf('Mean deviation of NN pred from best ff: %g\n', mean_dev);
    if ~isempty(mean_improve)
        mean_improve = sum(mean_improve) / length(mean_improve);
    else
        mean_improve = 0;
    end
    fprintf('Mean improvement ratio by NN: %g\n', mean_improve);

    %save results
    res = [res [mean_best_val m_epoch m_time mean_hits m_phits mean_dev mean_improve]'];

	% score = 0.4 * m_phits + 0.2 * mean_improve + 0.1 * mean_dev + 
	%score = (res(1, i) + res(4, i) + res(5, i)) / (1 + res(2, i) + res(6, i) + res(7, i));
	%fprintf('Final score: %g\n', score);
    plot(mean_ff);
end

% calc scores
%resmax = max(res')';
%normalize results
nres = (res - repmat(min(res')', 1, exp_num)) ./ repmat(max(res')', 1, exp_num);
%print scores
scores = [];
disp('-------------------------Scores---------------------------------');
for i = 1:exp_num
	%scores = [scores (nres(5, i) + 0.5 * nres(1, i)) / (1 + 0.3 * nres(2, i) + 0.7 * nres(6, i) + 0.9 * nres(7, i))];
	scores = [scores (nres(5, i) + 0.5 * nres(4, i) + 0.5 * revest(nres(1, i)) + 0.7 * revest(nres(6, i)) + 0.9 * revest(nres(7, i)))];
	%fprintf('Experiment %d score: %g\n', i, score);
end
%normalize scores
scores = scores / max(scores);
disp('Final scores:');
disp(scores);

%display best graph in bold
[unused, ind] = sort(scores, 'descend');
plot(graphs{find(scores, max(scores))}, 'LineWidth', 3);

hold off
disp('==================================================================================================');

function [y] = revest(x)
y = (2/(1 + x)) - 1;