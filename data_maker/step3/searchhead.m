%SEARCHHEAD Head search for head's radius
%
%   default parameters
%     A:      Binary matrix for labels of head points
%     i0:     Head points for x axis
%     j0:     Head points for y axis
%     radius: Search radius from head points

%   Example:
%     searchhead(A , i0, j0, radius)

%   Copyright Ling Bao.
%   July 12, 2017
function [PoswithHead] = searchhead(A , i0, j0, radius)
    PoswithHead = [];
    for i = max(1, i0 - radius):min(size(A, 1), i0 + radius)
        for j = max(1, j0 - radius):min(size(A, 2), j0 + radius)
            if A(i, j) == 1
                if distance(i, j, i0, j0) <= radius
                  PoswithHead = cat(1, PoswithHead, [i, j]);
                end
            end
        end
    end
end

