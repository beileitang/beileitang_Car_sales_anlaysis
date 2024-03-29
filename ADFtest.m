
function[stat] = ADFtest(rtn, p)
Y = rtn(p+1:end); 

X = [rtn(p: end-1)]; 

for i = 1:p-1
   X = [X, rtn(p+1-i: end-i)-rtn(p-i: end-i-1)];  
end

if model==2
   X = [ones(length(Y),1), X];  
elseif model==3
   X = [ones(length(Y),1), (1:length(Y))', X];  
end

para_hat = (X'*X)\(X'*Y); 

res = Y - X*para_hat; 
sig2 = mean(res.^2); 
COV = sig2*inv(X'*X); 

if model==1
    stat = (para_hat(1)-1)/sqrt(COV(1,1)); 
elseif model==2
    stat = (para_hat(2)-1)/sqrt(COV(2,2)); 
elseif model==3
    stat = (para_hat(3)-1)/sqrt(COV(3,3)); 
end
end