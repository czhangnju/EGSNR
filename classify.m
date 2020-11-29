function [pred] = classify(x, D, labels, imgsize, gamma)

num_class = length(unique(labels));
rec = D*x;
for i=1:num_class
    pos = find(labels==i);
    D_i = D(:, pos);
    x_i = x(pos);
    error = rec - D_i*x_i;
    s = svd(reshape(error,imgsize), 'econ');
    e(i) = MCP(s, 1, gamma);  
end
ind = find(e==min(e));
pred = ind(1);
end


