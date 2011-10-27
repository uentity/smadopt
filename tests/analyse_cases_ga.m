function [res, graphs, scores] = analyse_cases_ga(root_dir, exp_templ, which_exp, do_plot)
if nargin < 3
	which_exp = 1;
end
if nargin < 4
	do_plot = true;
end
exp_num = 1;

% calc cnum
exps = dir(strcat(root_dir, '/', exp_templ, '*'));
cnum = length(exps);
res = [];
graphs = cell(1, exp_num);
if do_plot
	figure;
	hold on
end

disp('==================================================================================================');
a = cell(1, cnum);
%read data
%mean_best = [];
min_rnum = 0;
mean_best_val = 0;
m_epoch = 0;
m_time = 0;
for j=1:cnum
	fname = strcat(root_dir, '/', sprintf('%s%d', exp_templ, j), '/', sprintf('_%d_ga_stat.txt',  which_exp));
	a{j} = dlmread(fname);
	%size(a{j})
	rnum = size(a{j});
	% check for octave
	if rnum(2) > 6
		a{j} = a{j}(1:end, end-5:end);
	end
	rnum = rnum(1);
	if j == 1 || rnum < min_rnum
		min_rnum = rnum;
	end
	mean_best_val = mean_best_val + min(a{j}(1:end, 3));
	m_epoch = m_epoch + rnum;
	m_time = m_time + a{j}(end, end);
end
mean_best_val = mean_best_val / cnum;
m_epoch = m_epoch / cnum;
m_time = m_time / cnum;

%calc mean graph of best fitness
mean_ff = zeros(min_rnum, 1);
for j = 1:cnum
	mean_ff = mean_ff + interp1((1:length(a{j}(1:end, 3)))', a{j}(1:end, 3), (1:min_rnum)');
end
mean_ff = mean_ff ./ cnum;
graphs{1} = mean_ff;

%print results
fprintf('Clear GA results:\n');
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

%save results
res = [res [mean_best_val m_epoch m_time]'];
if do_plot
	plot(log10(mean_ff), 'LineWidth', 2);
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
	scores = [scores (revest(nres(1, i)) + revest(nres(2, i)) + revest(nres(3, i)))];
	%fprintf('Experiment %d score: %g\n', i, score);
end
%normalize scores
scores = scores / max(scores);
disp('Final scores:');
disp(scores);

if do_plot
	%display best graph in bold
	plot(log10(graphs{scores == max(scores)}), '-k', 'LineWidth', 3);
	grid('on');
	set(gca,'fontsize',12);
	xlabel('Iterations', 'fontsize', 12);
	ylabel('Objective function value', 'fontsize', 12);
	hold off
end

disp('==================================================================================================');

function [y] = revest(x)
y = (2/(1 + x)) - 1;