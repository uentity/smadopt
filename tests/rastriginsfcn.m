function scores = rastriginsfcn(pop)
%RASTRIGINSFCN Compute the "Rastrigin" function.

%   Copyright 2003-2004 The MathWorks, Inc.
%   $Revision: 1.3.4.1 $  $Date: 2004/08/20 19:50:22 $


    % pop = max(-5.12,min(5.12,pop));
    scores = 10.0 * size(pop,2) + sum(pop .^2 - 10.0 * cos(2 * pi .* pop),2);
  


