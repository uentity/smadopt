function [res] = time_analysis(graphs, stat, np, fcn_name)
% globals
ga_idx = 3;
algo = {'GA+NN', 'GA+NN+AL'};
algo_ru = {'ГА+НС', 'ГА+НС+ДО'};
algo_idx = [1 2];

cases = [3 6];
vars_n = [5 100];
do_tex_export = true;

res = cell(length(algo), length(cases));
fid = 0;
% iterate through cases
for c = 1:length(cases)
	%if do_tex_export
	%	fname = sprintf('%s_ta_%d.tex', fcn_name, vars_n(c));
	%	fid = fopen(fname, 'w');
	%end
	% iterate algorithms
	for a = 1:length(algo)
		fprintf('Time analysis for %s fcn, %s algo, vars num = %d\n', fcn_name, algo{a}, vars_n(c));
		%if do_tex_export
		%	% print multirow algo name
		%	fprintf(fid, '\\multirow{%d}{*}{%s}', length(np), algo_ru{a});
		%end
		% iterate throgh all np settings
		for p = 1:length(np)
			s = proc_algo(graphs{ga_idx, cases(c)}, graphs{algo_idx(a), cases(c)}, stat{algo_idx(a), cases(c)}, np(p));
			res{a, c} = [res{a, c}; s];
			% display
			[h m sec] = expand_time(s(3));
			[h1 m1 sec1] = expand_time(s(4));
			fprintf('Np = %d: I = %d, Ih = %d, th = %g (%g h %g min %g s), tm = %g (%g h %g min %g s)\n', ...
				np(p), s(1), s(2), s(3), h, m, sec, s(4), h1, m1, sec1);
			%if do_tex_export
			%	tex_write(fid, np(p), s, p == length(np));
			%end
		end
		fprintf('\n');
    end
    %if do_tex_export
    %    fclose(fid);
    %end
end

if do_tex_export
	% iterate through cases
	for c = 1:length(cases)
		fname = sprintf('%s_ta_%d.tex', fcn_name, vars_n(c));
		fid = fopen(fname, 'w');
		% iterate algorithms
		for a = 1:length(algo)
			s = res{a, c};
			% calc real rows number (where t_M^* > 0
			rows_n = length(find(s(1:end, 4) > 0));
			if rows_n == 0 continue; end;
			% print multirow algo name
			fprintf(fid, '\\multirow{%d}{*}{%s}', rows_n, algo_ru{a});
			% iterate through Np settings
			for p = 1:length(np)
				% print multirow I, Ih and th values
				if p == 1
					% I Ih
					fprintf(fid, ' & \\multirow{%d}{*}{%d} & \\multirow{%d}{*}{%d}', rows_n, s(p, 1), rows_n, s(p, 2));
					% th
					[h mn sec] = expand_time(s(p, 3));
					fprintf(fid, ' & \\multirow{%d}{*}{%d:%d:%.2f}', rows_n, h, mn, sec);
				else
					fprintf(fid, ' & & &');
				end
				% print Np, t_m
				[h mn sec] = expand_time(s(p, 4));
				fprintf(fid, ' & %d & %d:%.2f \\\\', np(p), mn, sec);
				% print full or short hline
				if p == length(np)
					fprintf(fid, ' \\hline\n');
				else
					fprintf(fid, ' \\cline{5-6}\n');
				end
            end
        end
	fclose(fid);
    end
end

function r = proc_algo(std_g, alg_g, stat, np)
% calc th
th = 0;
if(stat(2) > 0)
	th = stat(3)/stat(2);
end
% calc Ih
Ih = 0;
i = find(alg_g < std_g(end));
if ~isempty(i)
	Ih = i(1);
end
% calc tm
I = length(std_g);
tm = th*Ih/(np*(I - Ih));
r = [I Ih th tm];

function tex_write(fid, p, stat, hline)
[h mn sec] = expand_time(stat(3));
[h1 mn1 sec1] = expand_time(stat(4));
if(stat(4) > 0)
    fprintf(fid, ' & %d & %d & %d & %d:%d:%.2f & %d:%.2f \\\\ ', p, stat(1), stat(2), h, mn, sec, mn1, sec1);
    if(hline)
        fprintf(fid, '\\hline\n');
    else
        fprintf(fid, '\\cline{2-6}\n');
    end
end

