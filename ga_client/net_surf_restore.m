function net_surf_restore
y = load('y.txt');
z = load('z.txt');
ly = load('inputs.txt')';
t = load('targets.txt')';
bs = load('best_sol.txt');
ss = load('ss.txt');
c = load('c.txt');
learn = [ly t];
test = [y; z]';
mode = 1;
if size(ly, 2) > 2
    if mode == 1
        %concat all data points
        y = y';
        dp = [ly; y; c];
        m = mean(dp);
        [coeff tdp] = princomp(dp);
        %extract tly
        pos = size(ly, 1);
        tly = tdp(1:pos, :);
        pos = pos + 1;
        %extract test
        ty = tdp(pos:pos + size(y, 1) - 1, :);
        pos = pos + size(y, 1);
        %extract centers
        tc = tdp(pos:end, :);        
        %transform search samples
        ss = ss - repmat(m, size(ss, 1), 1);
        tss = ss * coeff;
        %transform best solutions
        tbs = (bs(1:size(m)) - m) * coeff;
        %plot everything
        plot_data([tly(:, 1:2) t], [ty(:, 1:2) z'], tc(:, 1:2), [tbs(1:2) bs(1, size(bs))], tss(:, 1:2));
    elseif mode == 2
        m = mean(ly);
        [coeff tly] = princomp(ly);
        %transform test
        y = y';
        y = y - repmat(m, size(y, 1), 1);
        ty = y * coeff;
        %transform centers
        %c = [c zeros(size(c, 1), 1)];
        c = c - repmat(m, size(c, 1), 1);
        tc = c * coeff;
        %transform search samples
        %ss = [ss zeros(size(ss, 1), 1)];
        ss = ss - repmat(m, size(ss, 1), 1);
        tss = ss * coeff;
        %transform best solutions
        tbs = (bs(1:size(m)) - m) * coeff;
        %plot everything
        plot_data([tly(:, 1:2) t], [ty(:, 1:2) z'], tc(:, 1:2), [tbs(1:2) bs(1, size(bs))], tss(:, 1:2));
    else
        m = mean(learn);
        [coeff tlearn] = princomp(learn);
        %transform test
        test = test - repmat(m, size(test, 1), 1);
        ttest = test * coeff;
        %transform centers
        %c = [c zeros(size(c, 1), 1)];
        c = c - repmat(m(1:size(c, 2)), size(c, 1), 1);
        tc = c * coeff(1:size(c, 2), 1:size(c, 2));
        %transform search samples
        %ss = [ss zeros(size(ss, 1), 1)];
        ss = ss - repmat(m(1:size(ss, 2)), size(ss, 1), 1);
        tss = ss * coeff(1:size(ss, 2), 1:size(ss, 2));
        %transform best solutions
        tbs = (bs - m) * coeff;
        %plot everything
        plot_data(tlearn(:, 1:3), ttest(:, 1:3), tc, tbs, tss);
    end
else
    plot_data(learn, test, c, bs, ss);
end

function plot_data(learn, test, c, bs, ss)
plot3(learn(:, 1), learn(:, 2), learn(:, 3), '.r');
hold on

% another 3d surf plot engine
%[XI, YI] = meshgrid(min(test(:, 1)) : 0.05 : max(test(:, 1)), min(test(:, 2)) : 0.05 : max(test(:, 2)));
%ZI = griddata(test(:, 1), test(:, 2), test(:, 3), XI, YI, 'cubic');
%surfl(XI, YI, ZI);

tri = delaunay(test(:, 1), test(:, 2));
trisurf(tri, test(:, 1), test(:, 2), test(:, 3));

shading interp;
grid on;

plot3(c(:, 1), c(:, 2), zeros(size(c, 1), 1), '*black');
plot3(bs(1, 1), bs(1, 2), 0, 'og', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [.49 1 .63], 'MarkerSize', 12);
plot3(ss(:, 1), ss(:, 2), zeros(size(ss, 1), 1), '.g');
hold off
