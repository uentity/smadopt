function [y] = rosenbrocksfcn(x)
r = size(x, 2);
y = [];
for i = 1:size(x, 1)
    yi = 0;
    for j = 1:(r - 1)
        t = x(i, j) * x(i, j);
        yi = yi + 100*(t - x(i, j + 1))*(t - x(i, j + 1)) + (1 - x(i, j))*(1 - x(i, j));
    end
    y = [y; yi];
end