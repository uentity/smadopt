function show_res()
p = load('t.txt');
f = load('f.txt');
centers = load('centers.txt');
ind = load('ind.txt');
real_c = load('real_c.txt');
ind = ind + 1;

%sequental colored cluster plot
%cm = colormap('lines');
cm = gmap40(size(centers, 1));
hold on
for i=1:size(centers, 1)
    ind_c = find(ind == i);
    plot3(p(ind_c, 1), p(ind_c, 2), f(ind_c), '.', 'MarkerSize', 14, 'MarkerEdgeColor', cm(i, :));
    plot3(centers(i, 1), centers(i, 2), zeros(1, 1), 'vk', 'LineWidth', 2, 'MarkerSize', 11, 'MarkerFaceColor', 'r');
end
plot3(real_c(:, 1), real_c(:, 2), zeros(size(real_c, 1), 1), 'ok', 'LineWidth', 3, 'MarkerSize', 12);
hold off

disp('Real centers:'); disp(real_c);
disp('Found centers:'); disp(centers);
[rm, unused] = size(real_c);
err = [];
for i=1:rm
    cur_c = real_c(i, :);
    [cm, unused] = size(centers);
    dist = repmat(cur_c, cm, 1) - centers;
    dist = sum((dist.*dist)');
    [unused, min_ind] = min(dist);
    err = [err; sqrt(dist(min_ind))];
%    sum_dist = sum_dist + sqrt(dist(min_ind));
%    sum_dist2 = sum_dist2 + dist(min_ind);
    centers(min_ind, :) = [];
    if(isempty(centers)) 
        break; 
    end;
end;

disp('Not found centers:'); disp(real_c(i+1:end, :));
disp('Redudant centers:'); disp(centers);
disp('Errors in distances:'); disp(err);
disp('Mean error:'); disp(mean(err));
disp('Std variance of error:'); disp(std(err));

%surface plot
%plot3(centers(:, 1), centers(:, 2), zeros(size(centers, 1), 1), '+r');
%hold on
%tri = delaunay(p(:, 1), p(:, 2));
%trisurf(tri, p(:, 1), p(:, 2), f);
%shading interp;
