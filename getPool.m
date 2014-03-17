function pool = getPool(lVecs, rVecs, edgesize)
    lSize = size(lVecs, 2);
    rSize = size(rVecs, 2);
    pool = zeros(lSize, rSize);
    for i = 1:lSize
        for j = 1:rSize
            pool(i, j) = pdist([lVecs(:,i)'; rVecs(:,j)'], 'cosine');
            % unscaledPool(i, j) = norm(lVecs(:,i) - rVecs(:,j));
        end
    end
end