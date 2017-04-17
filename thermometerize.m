function [tValues] = thermometerize(values, nLevels)
    if nargin < 2
        nLevels = 5;
    end

    dValues = discretize(values, nLevels);

    tValues = zeros(length(values), nLevels-1);
    for i=1:length(dValues)
        tValues(i, 1:dValues(i)-1) = 1;
    end

end