library(forecast)
library(here)

setwd("C://Users//Zhenning Pei//Desktop//UWAT//fourth_year//4B//afm423//report//423-ML//DM_test")

har <- read.csv("har_res.csv", header = TRUE, colClasses=c('character','numeric'))

bag_har <- read.csv("bag_har_res.csv", header = TRUE, colClasses=c('character','numeric'))

enet <- read.csv("enet_res.csv", header = TRUE, colClasses=c('character','numeric'))

nn1 <- read.csv("RV_Residuals_1layer.csv", header = TRUE, colClasses=c('character','numeric'))
nn2 <- read.csv("RV_Residuals_2layer.csv", header = TRUE, colClasses=c('character','numeric'))
nn3 <- read.csv("RV_Residuals_3layer.csv", header = TRUE, colClasses=c('character','numeric'))

har_bag = dm.test(har[["res"]], bag_har[["res"]])[['p.value']]
har_enet = dm.test(har[["res"]], enet[["res"]])[['p.value']]
har_nn = dm.test(har[["res"]], nn1[["res"]])[['p.value']]
enet_nn = dm.test(enet[["res"]], nn1[["res"]])[['p.value']]

nn1_nn2 = dm.test(nn1[["res"]], nn2[["res"]])[['p.value']]
nn2_nn3 = dm.test(nn3[["res"]], nn2[["res"]])[['p.value']]
nn1_nn3 = dm.test(nn3[["res"]], nn1[["res"]])[['p.value']]

enet_bh = dm.test(enet[["res"]], bag_har[["res"]])[['p.value']]

bh_nn = dm.test(bag_har[["res"]], nn1[["res"]])[['p.value']]
