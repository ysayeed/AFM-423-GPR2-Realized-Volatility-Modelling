# load libraries
library(sn)
library(PerformanceAnalytics)
library(car)
library(tseries)
library(forecast)
library(quantmod)
library(MASS)
library(scales)

#target = 'C:/Users/Reginald Tao/OneDrive - University of Waterloo/MY ULOO/4BB/AFM423/Proj/Code/423-ML-master/Data/SPX_Index_processed.csv' #path to processed csv file


target = './Data/SPX_Index_processed.csv' #path to processed csv file, setwd to 423-ML-master
#target = './Data/SPX_Index_LPSD.csv'

#my wd:  C:/Users/Reginald Tao/OneDrive - University of Waterloo/MY ULOO/4BB/AFM423/Proj/Code/423-ML-master
df = read.csv(file = target, row.names = 'X')
df.z = as.zoo(df)

chart.TimeSeries(df.z, lwd=1)



table.Stats(df.z)

#normal fit
qqPlot(df$Realized.Volatility,main='RV Normal QQPlot')
jarque.bera.test(df$Realized.Volatility)


#lognormal fit
fit.params <- fitdistr(df$Realized.Volatility, "lognormal")
h <- hist(df$Realized.Volatility, breaks=seq(-0.0000,max(df$Realized.Volatility)+0.001,by=0.0015),plot=FALSE)
barplot(rbind(dlnorm(head(h$breaks,length(h$density)),fit.params$estimate["meanlog"], fit.params$estimate["sdlog"]),h$density),
        col=c('black','lightblue'),beside=TRUE,legend.text=c("Fitted Lognormal","Empirical Density of RV"),
        names.arg=round(seq(0,max(df$Realized.Volatility),length.out=length(h$counts)),4),
        xlab='RV',ylab='Density',main='RV Density and Fitted Lognormal Density',ylim=c(0,120))
qqPlot(df$Realized.Volatility,dist='lnorm',main='RV Lognormal QQPlot')



#qqPlot(df$Realized.Volatility, distribution="t", df=4)

#sn.df.fit = sn.mple(y=df$Realized.Volatility)
#sn.df.fit
#qqPlot(df$Realized.Volatility, dist="sn", location=sn.df.fit$cp[1], scale=sn.df.fit$cp[2], shape=sn.df.fit$cp[3])

#st.df.fit = st.mple(y=df$Realized.Volatility)
#qqPlot(df$Realized.Volatility, dist="st", location=st.df.fit$dp[1], scale=st.df.fit$dp[2], 
#      shape=st.df.fit$dp[3], df=st.df.fit$dp[4])



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

