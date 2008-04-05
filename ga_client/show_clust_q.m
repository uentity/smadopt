function show_clust_q
ly = load('inputs.txt');
plot(ly(1, :), ly(2, :), '.r');
hold on
c = load('c.txt');
plot3(c(:, 1), c(:, 2), zeros(size(c, 1), 1), '*black');
hold off