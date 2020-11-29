function out = MCP(s, lambda, gamma)
    s = abs(s);
    n = length(s);
    out = 0;
    for i=1:n
        u = s(i);
        if u<=gamma*lambda
            out = out + lambda*u-0.5*u*u/gamma;
        else
            out = out + 0.5*gamma*lambda*lambda;
        end
    end
end
