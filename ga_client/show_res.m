function stat = show_res()
global centers ind;

average = true;
plot_2d = true;
plot_3d = true;
asteps = 10;

if average
    plot_2d = false;
    plot_3d = false;
end

%load data
p = load('t.txt');
f = load('f.txt');
real_c = load('real_c.txt');

%do data processing
if average
    all_err = [];
    all_res = [];
    for i = 1:asteps
        %run clustering
        !ga_client.exe 1 20
        %process experiment results
        [pres, err] = process_data(p, f, real_c)
        %save results
        all_err = [all_err; err];
        all_res = [all_res pres];
    end
    disp('%============================================================================%');
    %calc stat
    stat = [[mean(all_res(1, :)); mean(all_res(2, :)); mean(all_err)]
        [std(all_res(1, :)); std(all_res(2, :)); std(all_err)]];
    disp('Overall statistics:\n');
    disp(stat);
else
    process_data(p, f, real_c);
end
if plot_2d
    do_2d_plot(p, f, real_c, centers, ind);
end
if plot_3d
    do_3d_plot(p, f, real_c, centers, ind);
end
end

function [pres, err] = process_data(p, f, real_c)
%load results
centers = load('centers.txt');
ind = load('ind.txt');
ind = ind + 1;

disp('Real centers:'); disp(real_c);
disp('Found centers:'); disp(centers);
rm = size(real_c, 1);
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
%save results
pres = [size(centers, 1); rm - i; mean(err); std(err)]
disp('Not found centers:'); disp(pres(2));
disp('Redudant centers:'); disp(centers);
disp('Errors in distances:'); disp(err);
disp('Mean error:'); disp(pres(3));
disp('Std variance of error:'); disp(pres(4));
disp('%-----------------------------------------------------------------------------%');
end

function do_2d_plot(p, f, real_c, centers, ind)
%plot 2d
figure(1);
clf;
%sequental colored cluster plot
%cm = colormap('lines');
cm = gmap40(size(centers, 1));
hold on
for i=1:size(centers, 1)
    ind_c = find(ind == i);
    plot(p(ind_c, 1), p(ind_c, 2), '.', 'MarkerSize', 14, 'MarkerEdgeColor', cm(i, :));
    plot(centers(i, 1), centers(i, 2), 'vk', 'LineWidth', 2, 'MarkerSize', 11, 'MarkerFaceColor', 'r');
end
plot(real_c(:, 1), real_c(:, 2), 'ok', 'LineWidth', 2, 'MarkerSize', 12);
hold off
grid on
set(gca,'fontsize',14);
set(gca,'fontname','arial');
end

function do_3d_plot(p, f, real_c, centers, ind)
figure(2)
clf;
%sequental colored cluster plot
%cm = colormap('lines');
cm = gmap40(size(centers, 1));
hold on
for i=1:size(centers, 1)
    ind_c = find(ind == i);
    plot3(p(ind_c, 1), p(ind_c, 2), f(ind_c), '.', 'MarkerSize', 14, 'MarkerEdgeColor', cm(i, :));
    plot3(centers(i, 1), centers(i, 2), zeros(1, 1), 'vk', 'LineWidth', 2, 'MarkerSize', 11, 'MarkerFaceColor', 'r');
end
plot3(real_c(:, 1), real_c(:, 2), zeros(size(real_c, 1), 1), 'ok', 'LineWidth', 2, 'MarkerSize', 12);
hold off
grid on
set(gca,'fontsize',14);
set(gca,'fontname','arial');
end

%surface plot
%plot3(centers(:, 1), centers(:, 2), zeros(size(centers, 1), 1), '+r');
%hold on
%tri = delaunay(p(:, 1), p(:, 2));
%trisurf(tri, p(:, 1), p(:, 2), f);
%shading interp;
