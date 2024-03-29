function[stat,pvalue] = LBtest(Y, m)

% Y: input data series, m: number of lags to be included

T = length(Y);

ACFhat = zeros(m,1); 
for i=1:m
   ACFhat(i) = (sum((Y(i+1:end)-mean(Y)).*(Y(1:end-i)-mean(Y))))/(sum((Y-mean(Y)).^2)); 
end

stat = 0; 
for i=1:m
   stat = stat+T*(T+2)*ACFhat(i)^2/(T-i); 
end

pvalue = chi2cdf(stat,m,'upper');
end

