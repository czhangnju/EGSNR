function [x, iter] = EGSNR(D, labels, y, weight, imgsize, options)
% D: data matrix, each column is a sample
% labels: label vector
% y: test sample
alpha = options.alpha; beta = options.beta; delta = options.delta;
gamma1 = options.gamma1; gamma2 = options.gamma2;  gamma3 = options.gamma3; 

num_class = length(unique(labels));

eps = 1e-3;
Max_Iter = 200; % maximum iteration step
mu = 1;
mu_max = 1e3;
rho  = 1.01;
x_init = zeros(size(D,2),1);
L  = D*x_init;
S = y - L;
x  = x_init;
u = x;
w  = u2uc(u, labels, num_class);
theta = w;
H = diag(weight);
Z1 = zeros(prod(imgsize), 1);
Z2 = zeros(length(labels), 1);
Z3 = zeros(num_class, 1);
Z4 = zeros(num_class, 1);

for iter = 1:Max_Iter   
    g1  =  y - D*x - S + 1/mu*Z1;
    Lm  =  proximal_matrix(Tm(g1, imgsize), alpha/mu, gamma1);
    L   =  Lm(:);
    
   g2 = y-D*x-L+1/mu*Z1;
   S  = proximal_vector(g2, prod(imgsize), beta/mu, gamma2);
   S  = S(:);
    

    g3 = D'*(y-L-S+1/mu*Z1) + u - (1/mu)*Z2;
    x = (D'*D+eye(size(D,2)))\g3;


    r = (x + Z2/mu)/2;
    p = (Z3 - mu*w)/2/mu;
    for i=1:num_class      
        r_i = r(labels==i);
        u(labels==i)=shrink(r_i, p(i));        
    end
    u_tild = u2uc(u, labels, num_class);
    u_tild = u_tild(:);
     

    Aw = H'*H + eye(num_class);
    Bw = u_tild + H'*theta + 1/mu*(Z3+H'*Z4);
    w = Aw\Bw;
    

    g5 = weight.*w - 1/mu*Z4;
    theta = proximal_vector(g5, num_class, delta/mu, gamma3);
    theta = theta(:);
    
    Z1 = Z1 + mu*(y - D*x - L - S);
    Z2 = Z2 + mu*(x - u);
    Z3 = Z3 + mu*(u_tild - w);
    Z4 = Z4 + mu*(theta - weight.*w);
    
    mu = min(rho*mu, mu_max);
    T1 = norm(y-D*x-L-S, 'inf');
    T2 =  norm(u_tild-w, 'inf');
    T3 =  norm(x-u, 'inf');
    T4 =  norm(theta-weight.*w, 'inf');
    convergence = T1<eps && T2<eps && T3<eps && T4<eps ;
    if convergence
        break;
    end
    
end


end


function [cu] = u2uc(u, labels, classnum)
    cu=zeros(classnum,1);
    for i = 1:classnum     
        cu(i) = norm(u(labels==i));        
    end
    cu = cu(:);
end

function out = proximal_matrix(Y, lambda, gamma)
[U, S, V] = svd(Y, 'econ');
s = diag(S);
n = length(s);
for i=1:n
    u = s(i);
    if ( u<gamma && u>=lambda )
        u = (u-lambda)/(1-lambda/gamma);
    elseif (u<lambda)
        u = 0;
    else
        u = s(i);
    end
    s(i) = u;   
end
out = U*diag(s)*V';

end

function x = proximal_vector(d, n, lambda, gamma)
	z = gamma*lambda;
	if (gamma >  1)
        for i = 1:n
			u = abs(d(i));
			x1 = min(z,max(0,gamma*(u - lambda)/(gamma - 1.0)));
			x2 = max(z,u);  
			if (0.5*(x1 + x2 - 2*u)*(x1 - x2) + x1*(lambda - 0.5*x1/gamma) - 0.5*z*lambda < 0)
				x(i) = x1;
			else
				x(i) = x2;
            end
            x(i) = threecompare(d(i)>=0, x(i), -x(i));
        end
        
    elseif (gamma < 1)
		for i=1:n
			u = abs(d(i));
			v = gamma *(u - lambda)/(gamma  - 1);
            x1 = threecompare(abs(v) > abs(v - z), 0, z);
			x2 = max(z,u);
			if (0.5*(x1 + x2 - 2*u)*(x1 - x2) + x1*(lambda - 0.5*x1/gamma) - 0.5*z*lambda < 0)
				x(i) = x1;
			else
				x(i) = x2;
            end
            x(i) = threecompare(d(i)>= 0, x(i), -x(i));
        end
        
    else
		for i=1:n
			u = abs(d(i));
            x1 = threecompare(lambda>u, 0, z);
			x2 = max(z,u);
			if (0.5*(x1 + x2 - 2*u)*(x1 - x2) + x1*(lambda - 0.5*x1/gamma) - 0.5*z*lambda < 0)
				x(i) = x1;
			else
				x(i) = x2;
            end
            x(i) = threecompare(d(i)>= 0, x(i), -x(i));
        end
    end
    x = x(:);
end


function [us] = shrink(u, rho)

    us = max(1-rho/norm(u), 0)*u;

end

function out = threecompare(a, b, c)
    if (a)
        out = b;
    else
        out = c;
    end
end

function out = Tm(x, imgsize)
    out = reshape(x, imgsize);
end





