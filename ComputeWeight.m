function W = ComputeWeight(D, labels, y)

num_class = length(unique(labels));
W = zeros(num_class, 1);
err = W;
for i=1:num_class
    pos = find(labels==i);
    x_i = (D(:,pos)'*D(:,pos)+1e-3*eye(length(pos)))\(D(:,pos)'*y);
    y_r = D(:,pos)*x_i;
    err(i) = norm(y-y_r, 2);
end
W = (err - min(err))/(max(err)-min(err));

end

