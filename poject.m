

clear all;
load car;
y=car.Sale;
train=y(1:end-12);
test=y(end-11:end);
T=length(train);
plot(train)
set(gca,'XTick',1:96)
title('Monthly sale for car');
xlabel('Monthly');
ylabel('Monthly Sale');
 %chekc stationary 
 h =adftest(train)
train= train(2:end)-train(1:end-1);
y=y(2:end)-y(1:end-1);
% de seasonal by lag 12
train= train(13:end)-train(1:end-12);
y=y(13:end)-y(1:end-12);
%
h=adftest(train)
%subplot(2,1,1);
%autocorr(train);
%title('sample of ACF');
%subplot(2,1,2);
%parcorr(train);
%title('sample of PACF');

% use aic and bic to determine lags
LOGL = zeros(4,4); % Initialize
PQ = zeros(4,4);
for p = 1:4
    for q = 1:4
        mod = arima(p,0,q);
        [fit,~,logL] = estimate(mod,y,'Display','off');
        LOGL(p,q) = logL;
        PQ(p,q) = p+q;
     end
end

LOGL = reshape(LOGL,16,1);
PQ = reshape(PQ,16,1);
[~,bic] = aicbic(LOGL,PQ+1,83);
reshape(bic,4,4)
[aic,~] = aicbic(LOGL,PQ+1,83);
reshape(aic,4,4)
%get resuiduals
Mdl = arima(3,0,4);
EstMdl = estimate(Mdl,y);
[res,~,logL] = infer(EstMdl,y);

%test arch effect 
[stat, pvalue] = LBtest(res,12)
[stat, pvalue] = LBtest(res.^2,12)


stdr = res/sqrt(EstMdl.Variance);

figure
%subplot(2,2,1);
%plot(stdr);
%title('Standardized Residuals')
%subplot(2,2,2);
%histogram(stdr,10);
%title('Standardized Residuals')
%subplot(2,2,3);
%autocorr(stdr);
%subplot(2,2,4);
%parcorr(stdr);


%forecasting 


phi0= 2.7333;
phi1 = 0.082372; 
phi2 = 0.42465;
phi3 = -0.20891;



theta1= -0.92868;
theta2= -0.22067;
theta3= 0.37435;
theta4= -0.22501;

predicted = zeros(12,1); 

predicted(1,1) = phi0 + phi1*train(end)  + phi2*train(end-1)  + phi3*train(end-2);
                  - theta1*res(end-12)   - theta2*res(end-13)   - theta3*res(end-14)   - theta4*res(end-15);  

 predicted(2,1) = phi0 + phi2*train(end) + phi3*train(end-1);
                  -theta2*res(end-12)-theta3*res(end-13)-theta4*res(end-14);  
                  
predicted(3,1) = phi0+ phi3*train(end-12)-theta3*res(end-12)-theta4*res(end-13); 
predicted(4,1) = phi0-theta4*res(end-12);

for t=5:12
        predicted(t,1) = phi0 + phi1*predicted(t-1,1)+  phi1*predicted(t-2,1)+  phi1*predicted(t-3,1);
       

end


% 4. plotting


YMSE=mse(predicted);



plot(1:12, predicted, 'r', 1:12, y(84:end),'b--')
legend('predicted', 'realized')
title('predicted and realized values')

h1 = plot(84:95,y(84:95),'Color',[.7,.7,.7]);
hold on
h2 = plot(84:95,predicted,'b','LineWidth',2);
h3 = plot(84:95,predicted + 1.96*sqrt(YMSE),'r:',...
		'LineWidth',2);
plot(84:95,predicted - 1.96*sqrt(YMSE),'r:','LineWidth',2);
legend([h1 h2 h3],'Observed','Forecast',...
		'95% Confidence Interval','Location','NorthWest');
title(['Next 12 month forecasts monthly sales and Approximate 95% '...
			'Confidence Intervals'])
hold off


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


