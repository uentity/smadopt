[res_rbn, g_rbn, sc_rbn] = analyse_cases('rosenb_100/da', 'rbn_', 8);
print('-dpng', '-r600', strcat(prefix, '_ga_rbn.png'));
[res_ccn, g_ccn, sc_ccn] = analyse_cases('rosenb_100/da', 'ccn_', 8);
print('-dpng', '-r600', strcat(prefix, '_ga_ccn.png'));
[res_ga, g_ga, sc_ga] = analyse_cases_ga('rosenb_100', 'ga_', 1);
print('-dpng', '-r600', strcat(prefix, '_ga.png'));
b_rbn = find(sc_rbn == max(sc_rbn));
b_ccn = find(sc_ccn == max(sc_ccn));
b_ga = find(sc_ga == max(sc_ga));
figure;
hold on
grid on
plot([1:length(g_rbn{b_rbn})], g_rbn{b_rbn}, '--k', 'LineWidth',2);
plot([1:length(g_ccn{b_ccn})], g_ccn{b_ccn}, ':k', 'LineWidth', 2);
plot([1:length(g_ga{b_ga})], g_ga{b_ga}, '-k', 'LineWidth', 1);
hold off
set(gca,'fontsize',12);
xlabel('Итерации', 'fontsize', 12);
ylabel('Значение целевой функции', 'fontsize', 12);
legend('ГА+РБНС', 'ГА+РБНС+ДО', 'ГА');
print('-dpng', '-r600', strcat(prefix, '_compare.png'));
