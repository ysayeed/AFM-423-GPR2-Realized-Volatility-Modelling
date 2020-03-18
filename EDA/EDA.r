# load libraries
library(sn)
library(PerformanceAnalytics)
library(car)
library(tseries)
library(forecast)
library(quantmod)

target = 'C:/Users/Reginald Tao/OneDrive - University of Waterloo/MY ULOO/4BB/AFM423/Proj/Code/423-ML-master/Data/SPX_Index_processed.csv' #path to processed csv file

df = read.csv(file = target, row.names = 'X')
df.z = as.zoo(df)
lnorm_rv=rlnorm(length(df[,1]),mean(df[,1],sd(df[,1])))
chart.TimeSeries(df.z, lwd=1)

#hist(df$Realized.Volatility, breaks=100, col="slateblue",freq=F)

table.Stats(df.z)

hist(df$Realized.Volatility, breaks=100, col="slateblue",freq=F,main='Hist of RV Density and Fitted Lognormal Density')
fit.params <- fitdistr(df$Realized.Volatility, "lognormal")
hist(rlnorm(length(df$Realized.Volatility),fit.params$estimate["meanlog"], fit.params$estimate["sdlog"]),breaks=100,col=alpha('red',0.9),add=T,freq=F)
legend('topright',legend=c("RV Density", "Fitted Lognormal Density"),
       col=c("blue", "red"),pch=0:0)


qqPlot(df$Realized.Volatility,main='RV Normal QQPlot')
jarque.bera.test(df$Realized.Volatility)


#qqPlot(df$Realized.Volatility, distribution="t", df=4)

#sn.df.fit = sn.mple(y=df$Realized.Volatility)
#sn.df.fit
#qqPlot(df$Realized.Volatility, dist="sn", location=sn.df.fit$cp[1], scale=sn.df.fit$cp[2], shape=sn.df.fit$cp[3])

#st.df.fit = st.mple(y=df$Realized.Volatility)
#qqPlot(df$Realized.Volatility, dist="st", location=st.df.fit$dp[1], scale=st.df.fit$dp[2], 
#      shape=st.df.fit$dp[3], df=st.df.fit$dp[4])

qqPlot(df$Realized.Volatility,dist='lnorm',main='RV Lognormal QQPlot')


# use Box.test from stats package
Box.test(df, type="Ljung-Box", lag = 21)
#
# Volatility clustering
#

dataToPlot.z = merge(df.z, df.z^2)
colnames(dataToPlot.z) = c("Volatility","Variance")
plot(dataToPlot.z,col="blue", main="Vol and Var Comparison")

par(mfrow=c(2,1))
acf(df, lwd=2,ylim=c(0,1),main="ACF of Realized Vol")
acf(df^2, lwd=2,ylim=c(0,1),main="ACF of Realized Var")
par(mfrow=c(1,1))

