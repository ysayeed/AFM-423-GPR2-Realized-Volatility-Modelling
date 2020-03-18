# load libraries
library(sn)
library(PerformanceAnalytics)
library(car)
library(tseries)
library(forecast)
library(quantmod)

target = '' #path to processed csv file

df = read.csv(file = target, row.names = 'X')
df.z = as.zoo(df)

chart.TimeSeries(df.z, lwd=1)

hist(df$Realized.Volatility, breaks=25, col="slateblue")
table.Stats(df.z)

chart.Histogram(df.z, methods="add.normal")

qqPlot(df$Realized.Volatility)
jarque.bera.test(df$Realized.Volatility)


qqPlot(df$Realized.Volatility, distribution="t", df=4)

sn.df.fit = sn.mple(y=df$Realized.Volatility)
sn.df.fit
qqPlot(df$Realized.Volatility, dist="sn", location=sn.df.fit$cp[1], scale=sn.df.fit$cp[2], shape=sn.df.fit$cp[3])

st.df.fit = st.mple(y=df$Realized.Volatility)
qqPlot(df$Realized.Volatility, dist="st", location=st.df.fit$dp[1], scale=st.df.fit$dp[2], 
       shape=st.df.fit$dp[3], df=st.df.fit$dp[4])

par(mfrow=c(2,1))
Acf(df, lwd=2)
par(mfrow=c(1,1))

# use Box.test from stats package
Box.test(df, type="Ljung-Box", lag = 21)
#
# Volatility clustering
#

dataToPlot.z = merge(df.z, df.z^2)
colnames(dataToPlot.z) = c("Volatility","Volatility^2")
plot(dataToPlot.z,col="blue", main="Volatility")

par(mfrow=c(3,1))
Acf(df, lwd=2)
Acf(df^2, lwd=2)
par(mfrow=c(1,1))

