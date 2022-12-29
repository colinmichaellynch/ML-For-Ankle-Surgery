rm(list = ls())

library(haven)
library(ggplot2)
library(dplyr)
library(plyr)

setwd("C:/Users/user/Documents/Fiverr projects/Elbow Surgery/Ankle")

dataFrame1 = read_sav("2018.sav")  
dataFrame2 = read_sav("2019.sav")  
dataFrame3 = read_sav("2020.sav")  
dataFrame4 = read_sav("2017.sav")
dataFrame5 = read_sav("2015.sav")
dataFrame6 = read_sav("2016.sav")
#dataFrame7 = read_sav("2012.sav")
#dataFrame8 = read_sav("2013.sav")
#dataFrame9 = read_sav("2014.sav")
dataFrame10 = read_sav("2021.sav")

#nameList = Reduce(intersect, list(names(dataFrame1), names(dataFrame2), names(dataFrame3), names(dataFrame4), names(dataFrame5), names(dataFrame6), names(dataFrame7), names(dataFrame8), names(dataFrame9), names(dataFrame10)))

nameList = Reduce(intersect, list(names(dataFrame1), names(dataFrame2), names(dataFrame3), names(dataFrame4), names(dataFrame5), names(dataFrame6), names(dataFrame10)))

dataFrame1 = dataFrame1[, nameList]
dataFrame2 = dataFrame2[, nameList]
dataFrame3 = dataFrame3[, nameList]
dataFrame4 = dataFrame4[, nameList]
dataFrame5 = dataFrame5[, nameList]
dataFrame6 = dataFrame6[, nameList]
#dataFrame7 = dataFrame7[, nameList]
#dataFrame8 = dataFrame8[, nameList]
#dataFrame9 = dataFrame9[, nameList]
dataFrame10 = dataFrame10[, nameList]

dataFrameCombined = rbind.fill(dataFrame1, dataFrame2, dataFrame3, dataFrame4, dataFrame5, dataFrame6, dataFrame10)

dataFrameCombined$BMI = 703 * dataFrameCombined$WEIGHT / (dataFrameCombined$HEIGHT^2)
dataFrameCombined$AGE = as.numeric(gsub("+","",dataFrameCombined$AGE ,fixed = TRUE))

dataFrameCombined = dataFrameCombined %>% select(order(colnames(dataFrameCombined)))

predictorList = c("AGE", "BMI", "SEX", "INOUT", "DIABETES", "RACE_NEW", "SMOKE", "VENTILAT", "HXCOPD", "ASCITES", "HXCHF", "HYPERMED", "DIALYSIS", "DISCANCR", "STEROID", "TRANSFUS", "ASACLAS") 

predictorList2 = c("AGE", "BMI", "SEX", "INOUT", "DIABETES", "RACE_NEW", "SMOKE", "VENTILAT", "HXCOPD", "ASCITES", "HXCHF", "HYPERMED", "DIALYSIS", "DISCANCR", "STEROID", "TRANSFUS", "ASACLAS", "PRALBUM", "PRSODM", "PRBUN", "PRPTT", "PRINR", "PRWBC", "PRSGOT") 

adverseList = c("SUPINFEC", "WNDINFD", "ORGSPCSSI", "DEHIS", "OUPNEUMO", "REINTUB", "PULEMBOL", "URNINFEC", "CNSCVA", "CDARREST", "CDMI", "OTHBLEED", "OTHDVT", "OTHSYSEP", "RETURNOR", "READMISSION1", "TOTHLOS", "OPTIME")

dataFrameCombinedPredictors = dataFrameCombined[,predictorList]

dataFrameCombinedPredictors$DIABETES[dataFrameCombinedPredictors$DIABETES != "NO"] = "YES"
dataFrameCombinedPredictors$RACE_NEW[dataFrameCombinedPredictors$RACE_NEW != "White"] = "Non-White"

colSums(is.na(dataFrameCombinedPredictors))

dataFrameCombinedEvents = dataFrameCombined[,adverseList]
#cutoff = quantile(dataFrameCombinedEvents$OPTIME, c(.85)) 
cutoff = quantile(dataFrameCombinedEvents$OPTIME, c(1)) 

colSums(is.na(dataFrameCombinedEvents))

dataFrameCombinedEventsNumeric = data.frame(as.numeric(dataFrameCombinedEvents$SUPINFEC != "No Complication"), as.numeric(dataFrameCombinedEvents$WNDINFD != "No Complication"), as.numeric(dataFrameCombinedEvents$ORGSPCSSI != "No Complication"), as.numeric(dataFrameCombinedEvents$DEHIS != "No Complication"), as.numeric(dataFrameCombinedEvents$OUPNEUMO != "No Complication"), as.numeric(dataFrameCombinedEvents$REINTUB != "No Complication"), as.numeric(dataFrameCombinedEvents$PULEMBOL != "No Complication"), as.numeric(dataFrameCombinedEvents$URNINFEC != "No Complication"), as.numeric(dataFrameCombinedEvents$CNSCVA != "No Complication"), as.numeric(dataFrameCombinedEvents$CDARREST != "No Complication"), as.numeric(dataFrameCombinedEvents$CDMI != "No Complication"), as.numeric(dataFrameCombinedEvents$OTHBLEED != "No Complication") +as.numeric(dataFrameCombinedEvents$OTHDVT != "No Complication"), as.numeric(dataFrameCombinedEvents$OTHSYSEP != "No Complication"), as.numeric(dataFrameCombinedEvents$RETURNOR != "No"), as.numeric(dataFrameCombinedEvents$READMISSION1 != "No"), as.numeric(dataFrameCombinedEvents$TOTHLOS >2))

sums = colSums(dataFrameCombinedEventsNumeric)

vec = as.numeric(dataFrameCombinedEvents$SUPINFEC != "No Complication") + as.numeric(dataFrameCombinedEvents$WNDINFD != "No Complication") + as.numeric(dataFrameCombinedEvents$ORGSPCSSI != "No Complication") + as.numeric(dataFrameCombinedEvents$DEHIS != "No Complication")  + as.numeric(dataFrameCombinedEvents$OUPNEUMO != "No Complication") + as.numeric(dataFrameCombinedEvents$REINTUB != "No Complication") +as.numeric(dataFrameCombinedEvents$PULEMBOL != "No Complication") +as.numeric(dataFrameCombinedEvents$URNINFEC != "No Complication") +as.numeric(dataFrameCombinedEvents$CNSCVA != "No Complication") +as.numeric(dataFrameCombinedEvents$CDARREST != "No Complication") +as.numeric(dataFrameCombinedEvents$CDMI != "No Complication") +as.numeric(dataFrameCombinedEvents$OTHBLEED != "No Complication") +as.numeric(dataFrameCombinedEvents$OTHDVT != "No Complication") +as.numeric(dataFrameCombinedEvents$OTHSYSEP != "No Complication") +as.numeric(dataFrameCombinedEvents$RETURNOR != "No") +as.numeric(dataFrameCombinedEvents$READMISSION1 != "No") + as.numeric(dataFrameCombinedEvents$TOTHLOS >2) + as.numeric(dataFrameCombinedEvents$OPTIME >= cutoff)
vec = as.numeric(vec>0)
sum(vec)/nrow(dataFrameCombinedPredictors)

dataFrameCombinedPredictors$AdverseEvents = vec
dataFrameCombined = dataFrameCombinedPredictors
dataFrameCombined = dataFrameCombined %>% select(order(colnames(dataFrameCombined)))

colNames = colnames(dataFrameCombined)
finalData = subset(dataFrameCombined, select = colNames )
finalData = finalData %>% select(order(colnames(finalData)))
write.csv(finalData, "C:/Users/user/Documents/Fiverr projects/Elbow Surgery/Final Folder/CombinedDatasetAnkle.csv")

# demographics

quantile(finalData$AGE, c(.25, .50, .75)) 
quantile(finalData$BMI, c(.25, .50, .75)) 
sum(finalData$SEX=='male')
sum(finalData$SEX=='male')/nrow(finalData)
sum(finalData$RACE_NEW=='White')
sum(finalData$RACE_NEW=='White')/nrow(finalData)
sum(finalData$DIABETES=='YES')
sum(finalData$DIABETES=='YES')/nrow(finalData)
sum(finalData$HYPERMED=='Yes')
sum(finalData$HYPERMED=='Yes')/nrow(finalData)
sum(finalData$INOUT=='Inpatient')
sum(finalData$INOUT=='Inpatient')/nrow(finalData)

table(finalData$ASACLAS)
table(finalData$ASACLAS)/nrow(finalData)
