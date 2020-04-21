# load libraries
library(tseries)
library(caret)
library(glmnet)
library(elasticnet)
library(plotmo)
library(car)
library(zoo)
library(knitr)
target = './Data/SPX_Index_LPSD.csv' #path to processed csv file, setwd to 423-ML-master

#my wd:  C:/Users/Reginald Tao/OneDrive - University of Waterloo/MY ULOO/4BB/AFM423/Proj/Code/423-ML-master

df = read.csv(file = target, row.names = 'X')
df.z = as.zoo(df)

#create traind and test df with lag 1:30
timeSlices <- createTimeSlices(1:nrow(df),initialWindow = 30, horizon = 30, 
                               fixedWindow = TRUE)


fullSlices<- data.frame(matrix(unlist(timeSlices[[1]]),
                               nrow=length(timeSlices[[1]]), byrow=T))


fulldf=lapply(fullSlices,function(x){df[x,]})
fulldf=log(data.frame(matrix(unlist(fulldf), ncol=length(fulldf), byrow=T)))
#set.seed(423)
#train_ind=sample(nrow(fulldf), 2500)
dfTrain=head(fulldf,.9*nrow(fulldf))
dfTrain=dfTrain[order(as.numeric(row.names(dfTrain))), ]
dfTest=tail(fulldf,.1*nrow(fulldf))
dfTest=dfTest[order(as.numeric(row.names(dfTest))), ]
rownames(dfTrain) <- NULL
rownames(dfTest) <- NULL
colnames(dfTrain)=paste0('lag',c(0:29))
colnames(dfTest)=paste0('lag',c(0:29))


#train enet
t_grid = expand.grid(fraction=seq(0.1,1,by=0.1),
                     lambda=c(0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.5))
enetM <- train(lag0 ~ ., data = dfTrain,
               method = "enet",
               preProcess = c("center", "scale"),
               tuneGrid=t_grid,#tuneLength = 10,
               trControl = trainControl(method = "cv",number = 10))

get_best_result = function(caret_fit) { 
  best = which(rownames(caret_fit$results) == rownames(caret_fit$bestTune)) 
  best_result = caret_fit$results[best, ] 
  rownames(best_result) = NULL 
  best_result
}

#enet gives lasso as best model
get_best_result(enetM)
plot(enetM,xlab='alpha')
#print coef of enet (same as lasso)
predict.enet(enetM$finalModel, s=enetM$bestTune[1, "fraction"], 
             type="coef", mode="fraction")$coefficients

#10 least important factors. lag7 in there is interesting. long lags are in there. 
#lag29 is not, indicating far away dates still have an impact on RV
tail(rownames(varImp(enetM)$importance)[order(-varImp(enetM)$importance)],10)

head(rownames(varImp(enetM)$importance)[order(-varImp(enetM)$importance)],10)
varImpdf=data.frame('Order'=c(1:10),'Most Important'=head(rownames(varImp(enetM)$importance)[order(-varImp(enetM)$importance)],10),
                    'Least Important'=tail(rownames(varImp(enetM)$importance)[order(-varImp(enetM)$importance)],10))

kable(varImpdf,latex=T)
#train lasso
lassoM=glmnet(model.matrix(lag0~.,dfTrain)[,-1], dfTrain[,1])
plot_glmnet(lassoM, label=5,xvar='lambda')

lassoM=cv.glmnet(model.matrix(lag0~.,dfTrain)[,-1], dfTrain[,1])
plot(lassoM)

#prediction test
set.seed(423)

lasso_pred=predict(lassoM,newx=as.matrix(dfTest[, -1],ncol=30),
                   type='response',s='lambda.min')
lasso_pred_comp=data.frame(exp(dfTest[,1]),exp(lasso_pred))
rownames(lasso_pred_comp) <- NULL
lasso_pred_comp_sum=summary(abs((lasso_pred_comp[,1]-lasso_pred_comp[,2])/lasso_pred_comp[,1]))
lasso_pred_comp_sum

#standardized residual. should ~N(0,1)
res_lasso=lasso_pred_comp[,1]-lasso_pred_comp[,2]
std_res_lasso=res_lasso/sd(res_lasso)
#most std res are within +-2 sigma
plot(std_res_lasso)
abline(h=c(-1.96,1.96),col='red')


#dates where std residuals are >+-3 sigma
row.names(tail(df,.1*nrow(df)))[which(abs(std_res_lasso)>3)]

