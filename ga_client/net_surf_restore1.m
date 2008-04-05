function net_surf_restore1
x = load('x.txt');
y = load('y.txt');
z = load('z.txt');
ly = load('inputs.txt');
t = load('targets.txt');
surf(x, y, z);
shading interp
%colormap bone
hold on
plot3(ly(:, 1), ly(:, 2), t, '.r');
c = load('c.txt');
plot3(c(:, 1), c(:, 2), zeros(size(c, 1), 1), '*black');
bs = load('best_sol.txt');
plot3(bs(1, 1), bs(1, 2), bs(1, 3), 'og', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', [.49 1 .63], 'MarkerSize', 12);
ss = load('ss.txt');
plot3(ss(:, 1), ss(:, 2), zeros(size(ss, 1), 1), '.g');
hold off