function [a] = plot_all(rootd, do_export, rbn_tpl, ccn_tpl, ga_tpl, exp_num, cnum, cga_exp)
if nargin < 8
	cga_exp = 1;
end
if nargin < 7
	cnum = 4;
end
if nargin < 6
	exp_num = 8;
end
if nargin < 5
    ga_tpl = 'cga_';
end
if nargin < 4
    ccn_tpl = 'ccn_';
end
if nargin < 3
    rbn_tpl = 'rbn_';
end

% plot tuning
font_size = 14;
font_name = 'verdana';

a = cell(3, 3);
[res_rbn, g_rbn, sc_rbn] = analyse_cases(rootd, rbn_tpl, exp_num, cnum);
a{1, 1} = res_rbn; a{1, 2} = g_rbn; a{1, 3} = sc_rbn;
if do_export
    print('-dpng', '-r600', strcat(prefix, '_ga_rbn.png'));
end

[res_ccn, g_ccn, sc_ccn] = analyse_cases(rootd, ccn_tpl, exp_num, cnum);
a{2, 1} = res_ccn; a{2, 2} = g_ccn; a{2, 3} = sc_ccn;
if do_export
    print('-dpng', '-r600', strcat(prefix, '_ga_ccn.png'));
end

[res_ga, g_ga, sc_ga] = analyse_cases_ga(rootd, ga_tpl, cga_exp);
a{3, 1} = res_ga; a{3, 2} = g_ga; a{3, 3} = sc_ga;
if do_export
    print('-dpng', '-r600', strcat(prefix, '_ga.png'));
end

%plot comparison graph
b_rbn = find(sc_rbn == max(sc_rbn));
b_ccn = find(sc_ccn == max(sc_ccn));
b_ga = find(sc_ga == max(sc_ga));
figure;
semilogy(g_rbn{b_rbn}, '--k', 'LineWidth',2);
hold on
semilogy(g_ccn{b_ccn}, ':k', 'LineWidth', 2);
semilogy(g_ga{b_ga}, '-k', 'LineWidth', 1);
hold off
grid on
set(gca,'fontsize', font_size);
%set(gcf,'defaultLegendFontName', font_name);
xlabel('Итерации', 'fontsize', font_size, 'fontname', font_name);
ylabel('Значение целевой функции', 'fontsize', font_size, 'fontname', font_name);
set(gca, 'FontName', font_name);
legend('ГА+РБНС', 'ГА+РБНС+ДО', 'ГА');
if do_export
    print('-dpng', '-r600', strcat(prefix, '_compare.png'));
end
