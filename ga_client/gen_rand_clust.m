function [cs points f] = gen_rand_clust(clust_num, p_num)
rand('state',sum(100*clock));

%generate cluster centers
range = [-3 3];
%cs = gen_rastr_cent(clust_num, range);
cs = gen_rand_cent(clust_num, range);

randn('state',sum(100*clock));
points = [];
f = [];
for i = 1:clust_num
    dist = cs - repmat(cs(i, :), clust_num, 1);
    dist = dist.*dist;
    dist = sqrt(sum(dist'));
    mult = min(dist(find(dist > 0)))*0.25;
    %mult = 10;
    
    cur_cp = repmat(cs(i, :), p_num, 1);
    distr = randn(size(cur_cp))*mult;
    p = cur_cp + distr;
    points = [points; p];
    %calc f
    %cur_f = p - cur_cp;
    
    %simple parabola
    cur_f = sqrt(sum((distr.*distr)')');

    %gauss bell
    %cur_f = exp(sum((distr.*distr)')');
    
    %use rastrigin's function
    %cur_f = rastriginsfcn(p);
    
    f = [f; cur_f];
end
dlmwrite('t.txt', points, 'delimiter', ' ');
dlmwrite('f.txt', f, 'delimiter', ' ');
dlmwrite('real_c.txt', cs, 'delimiter', ' ');

%function for generating rundm centers
function [cs] = gen_rand_cent(clust_num, range)
%generate random centers
cs = rand(clust_num, 2);
cs = cs*(range(2) - range(1)) + range(1);

%function for generating centers in rastrigin's function minimums
function [cs] = gen_rastr_cent(clust_num, range)
x1 = [round(range(1)):1:round(range(2))];
x2 = x1;
len = min(length(x1), clust_num);
x1 = x1(randperm(len));
x2 = x2(randperm(len));
cs = [x1(1:clust_num)' x2(1:clust_num)'];