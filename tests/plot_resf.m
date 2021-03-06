function [stat, graphs] = plot_resf(root_dir, fun_t, exp_num, cnum, rbn_templ, ccn_templ, cga_templ, mlp_templ)
global iter_limit
iter_limit = 100;

if nargin < 5
    rbn_templ = 'rbn_';
end
if nargin < 6
    ccn_templ = 'ccn_';
end
if nargin < 7
    cga_templ = 'cga_';
end
if nargin < 8
    mlp_templ = 'mlp_';
end

do_plot = true;
do_export = true;
do_time_plot = false;
do_time_export = true;
do_tex_export = false;
lng = 1;

% 1 - normal for inclusion in A4 (disser)
% 2 - autoref
% 3 - presentation
plotmode = 1;

if do_plot == false
    do_export = false;
end

if do_time_plot == false
    do_time_export = false;
end

%exp_templ = {rbn_templ; ccn_templ; cga_templ; mlp_templ};
exp_templ = {rbn_templ; ccn_templ; cga_templ};
%exp_templ = {rbn_templ; mlp_templ; cga_templ};

cga_idx = 3;
params = [2 3 5 10 50 100];
cmap = ['b'; 'r'; 'k'; 'g'];
line_st = {'-'; '--'; '-'; '.-'};
line_w = [2 2 1 2];
font_nm = 'verdana';

% font_sz here denotes main font size on resulting figure
% it is passed to plotpdftex
% figwidth = width of printed figure in cm
% figscale - scale factor of line widths
font_sz = 17;
figwidth = 18.5;
figscale = 1.1;
if plotmode == 2
	font_sz = 20;
	figscale = 1.5;
elseif plotmode == 3
	font_sz = 20;
	figscale = 1.5;
end

% tune axes font so that legend on figure from laplot fits the border
font_sz_axes = font_sz * 1.6;

%algs = {'ГА+НС'; 'ГА+НС+ДО'; 'ГА'; 'ГА+НС (МП)'};
%algs = {'ГА+НС'; 'ГА+НС+ДО'; 'ГА'};
algs = {'ГА+НС'; 'ГА+НС (МП)'; 'ГА'};

%algs_label = {'GA+NN'; 'GA+NN+AL'; 'GA'; 'GA+NN (MP)'};
%algs_label = {'GA+NN'; 'GA+NN (MP)'; 'GA'};
%algs_label = {'ГА+НС'; 'ГА+НС+ДО'; 'ГА'};
algs_label = {'ГА+НС'; 'ГА+НС (МП)'; 'ГА'};

%timel_legend = {'GA+NN'; 'GA+NN+AL'; 'GA+NN (MP)'};
%timel_legend = {'ГА+НС'; 'ГА+НС+ДО'; 'ГА'};
timel_legend = {'ГА+НС'; 'ГА+НС (МП)'; 'ГА'};

param_num_str = 'Число параметров';
iterations_str = 'Итерации';
calc_time_str = 'Процессорное время, ч';

if lng == 1
	% english labels
	algs = {'GA+NN'; 'GA+NN+AL'; 'GA'};
	algs_label = {'GA+NN'; 'GA+NN+AL'; 'GA'};
	timel_legend = {'GA+NN'; 'GA+NN+AL'; 'GA'};

	param_num_str = 'Prarmeters number';
	iterations_str = 'Iterations';
	calc_time_str = 'Calculation time, hrs';
end

% cd(root_dir);
root_dir = strcat(root_dir, '/', fun_t);

disp('==================================================================================================');
graphs = cell(length(exp_templ), exp_num);
stat = cell(length(exp_templ), exp_num);
time_line = [];
fid = 0;
%hold on
for i=1:exp_num
    if i > 1
        disp('-------------------');
    end
    fprintf('Experiment %d results:\n', i);
    
    if do_plot
        %calc stats and plot graphs for experiment i
        figure(i);
        h = figure(i);
        set(h, 'Position', [100 300 800 600]);
    end
    
    %override iterations limit for rosenb_2 case
    %if strcmp(fun_t, 'rosenb') && i == 1
    %    iter_limit = 200;
    %else
    %    iter_limit = 100;
    %end
    
    if do_tex_export
        fid = fopen(strcat(fun_t, '_gann_stat_', sprintf('%d', params(i)), '.tex'), 'w');
    end
    
    %colormap('Lines');
    %cmap = colormap;
    %hold on
    xlim = 0;
    for t=1:length(exp_templ)       
        fprintf('Results for: %s\n', exp_templ{t});
        stat_i = []; graph_i = [];        
        if t ~= cga_idx
            [stat_i graph_i] = process_ga_nn(i, cnum, root_dir, exp_templ{t});
            fprintf('\n');
        else
            [stat_i graph_i] = process_ga(i, cnum, root_dir, exp_templ{t});
        end        
        graphs{t, i} = graph_i;
        stat{t, i} = stat_i;
        
        if xlim == 0 || length(graph_i) < xlim
            xlim = length(graph_i);
        end
        
        if do_plot
            %plot off-line performance graph
			% special tuning for rastrigins fcn with 100 params
			if strcmp(fun_t, 'rastr') && params(i) == 100
				plot(graph_i, line_st{t}, 'LineWidth', line_w(t), 'Color', cmap(t));
			else
				semilogy(graph_i, line_st{t}, 'LineWidth', line_w(t), 'Color', cmap(t));
			end
            hold on
        end
        
        if do_tex_export
            tex_write(fid, algs(t), stat_i);
        end
    end
    if do_plot
        %format figure
        set(gca,'fontsize', font_sz_axes);
		set(gca,'fontname', font_nm);
        set(gca, 'xlim', [1 100], 'xtick', [1 20 40 60 80 100]);
        xlabel(strcat('\textbf{', iterations_str, '}'), 'fontsize', font_sz, 'fontname', font_nm);
		ylbl = '$f';
		if strcmp(fun_t, 'rastr')
			ylbl = strcat(ylbl, '_{\text{Ras}}');
			%if params(i) ~= 100
			%	ylbl = strcat('\log_{10} ', ylbl);
			%end
		elseif strcmp(fun_t, 'rosenb')
			ylbl = strcat(ylbl, '_{\text{Ros}}');
		end
        ylabel(strcat(ylbl, '(\mathbf x)$'), 'fontsize', font_sz, 'fontname', font_nm);
        title(sprintf(strcat('\\textbf{{', param_num_str, '}} $n$ = %d'), params(i)), 'fontsize', font_sz, 'fontname', font_nm);
        lh = legend(algs_label);
		%set(lh, 'FontSize', font_sz);
		if strcmp(fun_t, 'rastr') && params(i) == 5
			set(lh, 'Location', 'SouthWest');
		else
			set(lh, 'Location', 'NorthEast');
		end
        %legend('GA+NN', 'GA+NN+AL', 'GA');
        %set(gca, 'XLim', [1 xlim]);
        hold off
        grid('on');
    end
    
    if do_export
        fig_name = strcat(fun_t, '_gann_perf_');
		for t = 1:length(exp_templ)
			if exp_templ{t} == mlp_templ
				fig_name = strcat(fig_name, 'mlp_');
				break;
			end
		end
		fig_name = strcat(fig_name, sprintf('%d', params(i)));
		if plotmode == 2
			fig_name = strcat(fig_name, '_aref');
		elseif plotmode == 3
			fig_name = strcat(fig_name, '_report');
		end
		if lng == 1
			fig_name = strcat(fig_name, '_en');
		end
		% add extension
		fig_name = strcat(fig_name, '.pdf');
		% do plotting
		plotpdftex(h, fig_name, [font_sz figwidth figscale]);
        %print('-dpdf', fig_name);
        %unix(['epstopdf ./' fig_name]);
        %unix(['rm -f ' fig_name]);
    end
    
    if do_tex_export
        fclose(fid);
    end
    
    % fill timeline
    % collect times for hybrid algs
    time_slice = [];
    for t = 1:length(exp_templ)
        %if t ~= cga_idx
        time_slice = [time_slice stat{t, i}(3)];
        %end
    end
    time_line = [time_line; time_slice];
    %time_line = [time_line; [stat{1, i}(3) stat{2, i}(3)]];
end

if do_time_plot
    figure(exp_num + 1);
    h = figure(exp_num + 1);
    set(h, 'Position', [100 300 800 600]);
    
    time_line = time_line ./ 3600;
    for i = 1:size(time_line, 2)
        if i == cga_idx continue; end;
        plot([1:size(time_line, 1)], time_line(1:end, i), line_st{i}, 'LineWidth', line_w(i), 'Color', cmap(i));
        if i == 1
            hold on
        end
    end
    %plot([1:size(time_line, 1)], time_line(1:end, 1), 'b', 'LineWidth', 2);
    %hold on
    %plot([1:size(time_line, 1)], time_line(1:end, 2), '--r', 'LineWidth', 2);
    
    set(gca, 'XTick', [1:length(params)], 'XTickLabel', params);
    set(gca,'fontsize', font_sz_axes);
	set(gca,'fontname', font_nm);
    xlabel(strcat('\textbf{', param_num_str, '}'), 'fontsize', font_sz, 'fontname', font_nm);
    ylabel(stract('\textbf{', calc_time_str, '}'), 'fontsize', font_sz, 'fontname', font_nm);
    %title(sprintf('%d params', params(i)), 'fontsize', font_sz);
    lh = legend(timel_legend);
	%set(lh, 'FontSize', 23, 'Location', 'NorthWest');
    %legend('GA+NN', 'GA+NN+AL');
    grid on
    hold off
    
    if do_time_export
        fig_name = strcat(fun_t, '_gann_time_plot');
		if plotmode == 2
			fig_name = strcat(fig_name, '_aref');
		elseif plotmode == 3
			fig_name = strcat(fig_name, '_report');
		end
		if lng == 1
			fig_name = strcat(fig_name, '_en');
		end
		% add extension
		fig_name = strcat(fig_name, '.pdf');
		% do plotting
		plotpdftex(h, fig_name, [font_sz figwidth figscale]);
        %print('-dpdf', fig_name);
    end
end

disp('==================================================================================================');

%--------------------------------------------------------------------------
function [stat, graph] = process_ga_nn(exp_num, cnum, root_dir, templ)
global iter_limit
%prepare stat data
a = cell(1, cnum);
mean_hits = 0;
mean_dev = 0;
mean_best = [];
xlim = 0;
mean_improve = [];
mean_best_val = 0;
m_epoch = 0;
m_time = 0;
%read ga stat logs
for j=1:cnum
    fname = strcat(root_dir, '/', sprintf('%s%d', templ, j), '/', sprintf('_%d_ga_stat.txt', exp_num));
    a{j} = dlmread(fname);
    rnum = size(a{j});
    % check for octave
    %if rnum(2) > 8
    %    a{j} = a{j}(1:end, end-7:end);
    %end
    
    %apply iteration limit
    rnum = min([rnum(1) iter_limit]');
    a{j} = a{j}(1:rnum, 1:end);
    
    %xlim = xlim + rnum;
    if j == 1 || rnum > xlim
        xlim = rnum;
    end
    
    mean_hits = mean_hits + max(a{j}(1:end, 7));
    %tmp = (a{j}(1:end, 3) - a{j}(1:end, 6)) ./ a{j}(1:end, 3);
    %mean_dev = mean_dev + sqrt(sum(tmp.*tmp) / rnum);
    
    %filter NN outliers
    nn_pred = log(a{j}(1:end, 6));
    ind = find(nn_pred <= mean(nn_pred) + std(nn_pred)*2 & nn_pred >= mean(nn_pred) - std(nn_pred)*2);
    mean_dev = mean(abs(a{j}(ind, 3) - a{j}(ind, 6)));
    
    %calc mean improvement
    ind = find(a{j}(1:end, 7) ~= 0);
    ind = ind(find(ind ~= 1));
    if ~isempty(ind)
        mean_improve = [mean_improve; a{j}(ind, 3) ./ a{j}(ind - 1, 3)];
    end
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
%xlim = round(xlim / cnum);

%calc mean graph of best fitness
graph = mean_graph(a, cnum, xlim);

%calc % of NN hits
m_phits = mean_hits * 100 / m_epoch;

%print results
fprintf('Mean best ff achieved: %g\n', mean_best_val);
fprintf('Mean epochs: %g\n', m_epoch);
%expand time
[h mn sec] = expand_time(m_time);
fprintf('Mean optimization time: %g sec (%g h %g min %g sec)\n', m_time, h, mn, sec);
fprintf('Mean NN hits: %g\n', mean_hits);
fprintf('Mean %% of NN hits: %g%%\n', m_phits);
fprintf('Mean deviation of NN prediction from best ff: %g\n', mean_dev);
if ~isempty(mean_improve)
    mean_improve = sum(mean_improve) / length(mean_improve);
else
    mean_improve = 0;
end
fprintf('Mean improvement ratio by NN: %g\n', mean_improve);

%save statistics
stat = [mean_best_val m_epoch m_time mean_hits m_phits mean_dev mean_improve]';

%--------------------------------------------------------------------------
function [stat, graph] = process_ga(exp_num, cnum, root_dir, templ)
global iter_limit
a = cell(1, cnum);
%read data
mean_best = [];
xlim = 0;
mean_best_val = 0;
m_epoch = 0;
m_time = 0;
for j=1:cnum
    fname = strcat(root_dir, '/', sprintf('%s%d', templ, j), '/', sprintf('_%d_ga_stat.txt', exp_num));
    a{j} = dlmread(fname);
    %size(a{j})
    rnum = size(a{j});
    % check for octave
    %if rnum(2) > 6
    %    a{j} = a{j}(1:end, end-5:end);
    %end
    
    %apply iteration limit
    rnum = min([rnum(1) iter_limit]');
    a{j} = a{j}(1:rnum, 1:end);

    %xlim = xlim + rnum;
    if j == 1 || rnum > xlim
        xlim = rnum;
    end
    
    mean_best_val = mean_best_val + min(a{j}(1:end, 3));
    m_epoch = m_epoch + rnum;
    m_time = m_time + a{j}(end, end);
end
mean_best_val = mean_best_val / cnum;
m_epoch = m_epoch / cnum;
m_time = m_time / cnum;
%xlim = round(xlim / cnum);

%calc mean graph of best fitness
graph = mean_graph(a, cnum, xlim);

%print results
fprintf('Mean best ff achieved: %g\n', mean_best_val);
fprintf('Mean epochs: %g\n', m_epoch);
%expand time
[h mn sec] = expand_time(m_time);
fprintf('Mean optimization time: %g sec (%g h %g min %g sec)\n', m_time, h, mn, sec);

%save results
stat = [mean_best_val m_epoch m_time]';

function [mean_ff] = mean_graph(a, cnum, xlim)
%calc mean graph of best fitness
mean_ff = zeros(xlim, 1);
for j = 1:cnum
    gl = length(a{j}(1:end, 3));
    if gl ~= xlim
        delta = (gl - 1) / (xlim - 1);
        mean_ff = mean_ff + interp1([1:length(a{j}(1:end, 3))]', a{j}(1:end, 3), (1:delta:gl)', 'linear');
    else
        mean_ff = mean_ff + a{j}(1:end, 3);
    end
end
mean_ff = mean_ff ./ cnum;

function [h mn sec] = expand_time(m_time)
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

function tex_write(fid, alg_name, stat)
[h mn sec] = expand_time(stat(3));
% first part of table
fmt = '%s & %d & %.2f & %d:%d:%.';
% if time is < 1 sec then use .4 precision for seconds, otherwise round to
% integer sec
if stat(3) < 1
	fmt = strcat(fmt, '3');
end
fmt = strcat(fmt, 'f & ');
if(length(stat) > 3)
	fmt = strcat(fmt, '%.2f & %.2f & %.2f \\\\ \\hline\n');
	fprintf(fid, fmt, alg_name{1}, stat(1), stat(2), h, mn, sec, stat(4), stat(5), stat(7));
    %fprintf(fid, '%s & %d & %.2f & %d:%d:%.f & %.2f & %.2f & %.2f \\\\ \\hline\n', alg_name{1}, stat(1), stat(2), h, mn, sec, stat(4), stat(5), stat(7));
else
	fmt = strcat(fmt, '--- & --- & --- \\\\ \\hline\n');
	fprintf(fid, fmt, alg_name{1}, stat(1), stat(2), h, mn, sec);
    %fprintf(fid, '%s & %d & %.2f & %d:%d:%.f & --- & --- & --- \\\\ \\hline\n', alg_name{1}, stat(1), stat(2), h, mn, sec);
end
