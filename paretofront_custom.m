function [index,score] = paretofront_custom(y)
n = size(y,1);
index = true(1,n);

for i = 1:n
    for j = 1:n
        if all(y(j,:) >= y(i,:)) && any(y(j,:) > y(i,:))
            index(i) = false;
            break;
        end
    end
end

score = y(index,:);
end