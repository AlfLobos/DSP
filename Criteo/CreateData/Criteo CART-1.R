install.packages('caret')
install.packages('caTools')
install.packages('rpart.plot')
install.packages('e1071')
install.packages('forcats')
install.packages("rattle")
install.packages("rpart.utils")
install.packages("reshape")
install.packages("DescTools")
install.packages("ineq")
install.packages("treeClust")
install.packages("dtplyr")

library(dplyr)
library(ggplot2)
library(caTools) # splits
library(rpart) # CART
library(rpart.plot) # CART plotting
library(caret) # cross validation
library(forcats)
library(rattle)
library(rpart.utils)
library(reshape)
library(DescTools)
library(ineq)
library(treeClust)
library(data.table)
library(dtplyr)

cpVals = data.frame(cp = seq(0, .04, by=.002))

cad$cat1 = as.factor(cad$cat1)
cad$cat2 = as.factor(cad$cat2)
cad$cat3 = as.factor(cad$cat3)
cad$cat4 = as.factor(cad$cat4)
cad$cat5 = as.factor(cad$cat5)
cad$cat6 = as.factor(cad$cat6)
cad$cat7 = as.factor(cad$cat7)
cad$cat8 = as.factor(cad$cat8)
cad$cat9 = as.factor(cad$cat9)
cad$click = as.factor(cad$click)
sample_cad$click = as.factor(sample_cad$click)


set.seed(213)
split <- sample.split(cad$click, 0.7)
sample_cad.train <- filter(cad, split == TRUE)
sample_cad.test <- filter(cad, split == FALSE)

cadDummy2 = cadDummy

train.cart <- train(click ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + cat9,
                    data = cadTrain,
                    method = "rpart",
                    parms=list(split = 'gini'),
                    tuneGrid = cpVals,
                    trControl = trainControl(method = "cv", number=10),
                    metric = "Accuracy")

mod2 = rpart(click ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + cat9,
             data = cadTrain, method="class", 
             parms=list(split = 'gini'),
             minbucket = 5, cp = 0.002)

prp(mod2, digits=3)

sample_cad = sample_n(cad, 500000)

#Histograms
hist(cad$cat3)
barplot(prop.table(table(cad$cat7)))
barplot(sort(prop.table(table(cad$cat7)), decreasing = TRUE))
barplot(table(cad$cat7))
barplot(sort(table(cad$cat7), decreasing = TRUE))
ggplot(mutate(cad, Category = fct_infreq(cad$cat3))) + geom_bar(aes(x = cad$cat3))
ggplot(cad$cat3) + geom_bar(aes(x = cad$cat3))

cat3$all_percent = cat3$all_count/sum(cat3$all_count)
cat3v2 = cat3[order(cat3$all_percent, decreasing = TRUE),]
sum(cat3v2$all_percent[1:450])

cat7$all_percent = cat7$all_count/sum(cat7$all_count)
cat7v2 = cat7[order(cat7$all_percent, decreasing = TRUE),]
sum(cat7v2$all_percent[1:1500])

#Campaign Splitting
cadTrain = cadDummy[which(cad$timestamp < 2066399),]
cadTest = cadDummy[which(cad$timestamp >=  2066399),]
cadTrain2 = cadDummy2[which(cad$timestamp < 2066399),]
cadTest2 = cadDummy2[which(cad$timestamp >=  2066399),]
trainResults = as.data.frame(table(cadTrain$campaign))
testResults = as.data.frame(table(cadTest$campaign))
Overall = merge(trainResults, testResults, by = "Var1", all = T)  

#Adjusting cat3 and cat7
cat3v2$value2 = cat3v2$value
cat7v2$value2 = cat7v2$value
cat3v2$value2[cat3v2$all_percent < 0.0018457] <- 0
cat7v2$value2[cat7v2$all_percent < 0.0009582204] <- 0

#Adjusting original Dataset
cadDummy= cad
cadDummy$cat3 <- ifelse(cadDummy$cat3 %in% cat3v2$value2, cadDummy$cat3, 00000000)
cadDummy$cat7 <- ifelse(cadDummy$cat7 %in% cat7v2$value2, cadDummy$cat7, 00000000)

cadDummy$cat1 = as.factor(cadDummy$cat1)
cadDummy$cat2 = as.factor(cadDummy$cat2)
cadDummy$cat3 = as.factor(cadDummy$cat3)
cadDummy$cat4 = as.factor(cadDummy$cat4)
cadDummy$cat5 = as.factor(cadDummy$cat5)
cadDummy$cat6 = as.factor(cadDummy$cat6)
cadDummy$cat7 = as.factor(cadDummy$cat7)
cadDummy$cat8 = as.factor(cadDummy$cat8)
cadDummy$cat9 = as.factor(cadDummy$cat9)
cadDummy$click = as.factor(cadDummy$click)
cadDummy$conversion = as.factor(cadDummy$conversion)

fancyRpartPlot(mod2)
rpart.subrules.table(mod2)
str(mod2)
leafClusters = rpart.lists(mod2)
#accessing first value
atest[[1]][1]

cadDummy2$cluster <- ifelse(cadDummy2$cat9 %in% cluster1$V1, 1, 0)
cadDummy2$cluster <- ifelse(cadDummy2$cat9 %in% cluster2$V1 &
                              cadDummy2$cat9 %in% cluster4$V1, 2, cadDummy2$cluster)
cadDummy2$cluster <- ifelse(cadDummy2$cat9 %in% cluster2$V1 &
                              cadDummy2$cat9 %in% cluster3$V1 &
                              cadDummy2$cat5 %in% cluster6$V1, 3, cadDummy2$cluster)
cadDummy2$cluster <- ifelse(cadDummy2$cat9 %in% cluster2$V1 &
                              cadDummy2$cat9 %in% cluster3$V1 &
                              cadDummy2$cat5 %in% cluster5$V1 &
                              cadDummy2$cat8 %in% cluster7$V1, 4, cadDummy2$cluster)
cadDummy2$cluster <- ifelse(cadDummy2$cat9 %in% cluster2$V1 &
                              cadDummy2$cat9 %in% cluster3$V1 &
                              cadDummy2$cat5 %in% cluster5$V1 &
                              cadDummy2$cat8 %in% cluster8$V1 &
                              cadDummy2$cat9 %in% cluster9$V1, 5, cadDummy2$cluster)
cadDummy2$cluster <- ifelse(cadDummy2$cat9 %in% cluster2$V1 &
                              cadDummy2$cat9 %in% cluster3$V1 &
                              cadDummy2$cat5 %in% cluster5$V1 &
                              cadDummy2$cat8 %in% cluster8$V1 &
                              cadDummy2$cat9 %in% cluster10$V1 &
                              cadDummy2$cat3 %in% cluster12$V1, 6, cadDummy2$cluster)
cadDummy2$cluster <- ifelse(cadDummy2$cat9 %in% cluster2$V1 &
                              cadDummy2$cat9 %in% cluster3$V1 &
                              cadDummy2$cat5 %in% cluster5$V1 &
                              cadDummy2$cat8 %in% cluster8$V1 &
                              cadDummy2$cat9 %in% cluster10$V1 &
                              cadDummy2$cat3 %in% cluster11$V1 &
                              cadDummy2$cat8 %in% cluster13$V1, 7, cadDummy2$cluster)
cadDummy2$cluster <- ifelse(cadDummy2$cat9 %in% cluster2$V1 &
                              cadDummy2$cat9 %in% cluster3$V1 &
                              cadDummy2$cat5 %in% cluster5$V1 &
                              cadDummy2$cat8 %in% cluster8$V1 &
                              cadDummy2$cat9 %in% cluster10$V1 &
                              cadDummy2$cat3 %in% cluster11$V1 &
                              cadDummy2$cat8 %in% cluster14$V1, 8, cadDummy2$cluster)

leafClusters[[1]][1] = as.factor(leafClusters[[1]][1])
cadDummy2$cat9 = as.character(cadDummy2$cat9)
cat9Cluster = leafClusters[[1]][1]

cluster1 <- as.data.frame(matrix(unlist(df[1]), ncol=length(df[1]), byrow=T))
cluster2 = as.data.frame(matrix(unlist(df[2]), ncol=length(df[2]), byrow=T))
cluster3 = as.data.frame(matrix(unlist(df[3]), ncol=length(df[3]), byrow=T))
cluster4 = as.data.frame(matrix(unlist(df[4]), ncol=length(df[4]), byrow=T))
cluster5 = as.data.frame(matrix(unlist(df[5]), ncol=length(df[5]), byrow=T))
cluster6 = as.data.frame(matrix(unlist(df[6]), ncol=length(df[6]), byrow=T))
cluster7 = as.data.frame(matrix(unlist(df[7]), ncol=length(df[7]), byrow=T))
cluster8 = as.data.frame(matrix(unlist(df[8]), ncol=length(df[8]), byrow=T))
cluster9 = as.data.frame(matrix(unlist(df[9]), ncol=length(df[9]), byrow=T))
cluster10 = as.data.frame(matrix(unlist(df[10]), ncol=length(df[10]), byrow=T))
cluster11 = as.data.frame(matrix(unlist(df[11]), ncol=length(df[11]), byrow=T))
cluster12 = as.data.frame(matrix(unlist(df[12]), ncol=length(df[12]), byrow=T))
cluster13 = as.data.frame(matrix(unlist(df[13]), ncol=length(df[13]), byrow=T))
cluster14 = as.data.frame(matrix(unlist(df[14]), ncol=length(df[14]), byrow=T))

df = do.call(rbind, leafClusters)
df = as.data.frame(t(as.data.frame()))

names = lapply(leafClusters, `[[`, 1)
data = setNames(lapply(leafClusters, `[`, -1), names)
require(data.table)
setDF(sapply(ll, function(x) setattr(transpose(x[-1L]), 'names', x[[1L]])))
  
count(cadTrain2, c("campaign", "cluster"))
Obs = cadTrain2 %>% count(campaign, cluster)
Obs = cadTrain2 %>% 
              group_by(campaign, cluster) %>% 
              summarise(count = length(cluster))
Clicks = cadTrain2 %>% 
              group_by(campaign, cluster) %>% 
              summarise(clicks = sum(click == 1))

Conversions = cadTrain2 %>% 
              group_by(campaign, cluster) %>% 
              summarise(conversions = sum(conversion == 1))

Final = cadTrain2 %>% 
  group_by(campaign, cluster) %>% 
  summarise(count = length(cluster),
            clicks = sum(click == 1),
            conversions = sum(conversion == 1))

Final2 = cadTest2 %>% 
  group_by(campaign, cluster) %>% 
  summarise(count = length(cluster),
            clicks = sum(click == 1),
            conversions = sum(conversion == 1))

Obs2 = cast(Obs, cluster ~ campaign)
Clicks2 = cast(Clicks, cluster ~ campaign)
Conversions2 = cast(Conversions, cluster ~ campaign)
Final$combined <- paste(Final$count, ",", Final$clicks, ",", Final$conversions)
Final2$combined <- paste(Final2$count, ",", Final2$clicks, ",", Final2$conversions)


TrainTable = cast(Final, cluster ~ campaign)
TestTable = cast(Final2, cluster ~ campaign)

out = predict(mod2)
pred.response <- colnames(out)[max.col(out, ties.method = c("random"))]
1 - mean(cadTrain$click != pred.response)

set.seed(145)
dummySplit =  sample.split(cadTrain$click, SplitRatio = .75) 
train3 = subset(cadTrain, dummySplit == TRUE)
test3 = subset(cadTrain, dummySplit == FALSE)

#trying different levels of depth

mod0 = rpart(click ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + cat9,
             data = train3, method="class", 
             parms=list(split = 'gini'),
             minbucket = 5, cp = 0)

mod0_001 = rpart(click ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + cat9,
             data = train3, method="class", 
             parms=list(split = 'gini'),
             minbucket = 5, cp = 0.001)

mod0_002 = rpart(click ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + cat9,
                 data = train3, method="class", 
                 parms=list(split = 'gini'),
                 minbucket = 5, cp = 0.002)

mod0_003 = rpart(click ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + cat9,
                 data = train3, method="class", 
                 parms=list(split = 'gini'),
                 minbucket = 5, cp = 0.003)

mod0_004 = rpart(click ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + cat9,
                 data = train3, method="class", 
                 parms=list(split = 'gini'),
                 minbucket = 5, cp = 0.004)

mod0_005 = rpart(click ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + cat9,
                 data = train3, method="class", 
                 parms=list(split = 'gini'),
                 minbucket = 5, cp = 0.005)

mod0_006 = rpart(click ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + cat9,
                 data = train3, method="class", 
                 parms=list(split = 'gini'),
                 minbucket = 5, cp = 0.006)

mod0_007 = rpart(click ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + cat9,
                 data = train3, method="class", 
                 parms=list(split = 'gini'),
                 minbucket = 5, cp = 0.007)

mod0_008 = rpart(click ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + cat9,
                 data = train3, method="class", 
                 parms=list(split = 'gini'),
                 minbucket = 5, cp = 0.008)

mod0_009 = rpart(click ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + cat9,
                 data = train3, method="class", 
                 parms=list(split = 'gini'),
                 minbucket = 5, cp = 0.009)

mod0_01 = rpart(click ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + cat9,
                 data = train3, method="class", 
                 parms=list(split = 'gini'),
                 minbucket = 5, cp = 0.01)

out0 = predict(mod0)
pred.response0 <- colnames(out0)[max.col(out0, ties.method = c("random"))]
1 - mean(train3$click != pred.response0)

out0test = predict(mod0, test3)
pred.response0test <- colnames(out0test)[max.col(out0test, ties.method = c("random"))]
1 - mean(test3$click != pred.response0test)

out001 = predict(mod0_001)
pred.response001 <- colnames(out001)[max.col(out001, ties.method = c("random"))]
1 - mean(train3$click != pred.response001)

out001test = predict(mod0_001, test3)
pred.response001test <- colnames(out001test)[max.col(out001test, ties.method = c("random"))]
1 - mean(test3$click != pred.response001test)

out002 = predict(mod0_002)
pred.response002 <- colnames(out002)[max.col(out002, ties.method = c("random"))]
1 - mean(train3$click != pred.response002)

out002test = predict(mod0_002, test3)
pred.response002test <- colnames(out002test)[max.col(out002test, ties.method = c("random"))]
1 - mean(test3$click != pred.response002test)

out003 = predict(mod0_003)
pred.response003 <- colnames(out003)[max.col(out003, ties.method = c("random"))]
1 - mean(train3$click != pred.response003)

out003test = predict(mod0_003, test3)
pred.response003test <- colnames(out003test)[max.col(out003test, ties.method = c("random"))]
1 - mean(test3$click != pred.response003test)

out004 = predict(mod0_004)
pred.response004 <- colnames(out004)[max.col(out004, ties.method = c("random"))]
1 - mean(train3$click != pred.response004)

out005 = predict(mod0_005)
pred.response005 <- colnames(out005)[max.col(out005, ties.method = c("random"))]
1 - mean(train3$click != pred.response005)

out005test = predict(mod0_005, test3)
pred.response005test <- colnames(out005test)[max.col(out005test, ties.method = c("random"))]
1 - mean(test3$click != pred.response005test)

out006 = predict(mod0_006)
pred.response006 <- colnames(out006)[max.col(out006, ties.method = c("random"))]
1 - mean(train3$click != pred.response006)

out007 = predict(mod0_007)
pred.response007 <- colnames(out007)[max.col(out007, ties.method = c("random"))]
1 - mean(train3$click != pred.response007)

out008 = predict(mod0_008)
pred.response008 <- colnames(out008)[max.col(out008, ties.method = c("random"))]
1 - mean(train3$click != pred.response008)

out009 = predict(mod0_009)
pred.response009 <- colnames(out009)[max.col(out009, ties.method = c("random"))]
1 - mean(train3$click != pred.response009)

out01 = predict(mod0_01)
pred.response01 <- colnames(out01)[max.col(out01, ties.method = c("random"))]
1 - mean(train3$click != pred.response01)

#create vector
LogVectors = seq(-18.42068, -6.907755, length.out = 100)
LogVectors = as.data.frame(LogVectors)

mod_1 = rpart(click ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + cat9,
             data = train3, method="class", 
             parms=list(split = 'gini'),
             minbucket = 5, cp = 0.00000001000000744)

mod_2 = rpart(click ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + cat9,
              data = train3, method="class", 
              parms=list(split = 'gini'),
              minbucket = 5, cp = 0.00000001123324671)

mod_3 = rpart(click ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + cat9,
              data = train3, method="class", 
              parms=list(split = 'gini'),
              minbucket = 5, cp = 0.00000001261857377)

mod_4 = rpart(click ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + cat9,
              data = train3, method="class", 
              parms=list(split = 'gini'),
              minbucket = 5, cp = 0.00000001417475885)

mod_50 = rpart(click ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + cat9,
              data = train3, method="class", 
              parms=list(split = 'gini'),
              minbucket = 5, cp = 0.000002983647538)

mod_25 = rpart(click ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + cat9,
               data = train3, method="class", 
               parms=list(split = 'gini'),
               minbucket = 5, cp = 0.0000001629751666)

mod_13 = rpart(click ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + cat9,
               data = train3, method="class", 
               parms=list(split = 'gini'),
               minbucket = 5, cp = 0.0000000403701979)

#getting leaves
clusters51 = rpart.predict.leaves(mod51, cadTrain, type = "where")
clusters53 = rpart.predict.leaves(mod53, cadTrain, type = "where")
clusters55 = rpart.predict.leaves(mod55, cadTrain, type = "where")
clusters58 = rpart.predict.leaves(mod58, cadTrain, type = "where")
clusters62 = rpart.predict.leaves(mod62, cadTrain, type = "where")
clusters66 = rpart.predict.leaves(mod66, cadTrain, type = "where")
clusters70 = rpart.predict.leaves(mod70, cadTrain, type = "where")
clusters76 = rpart.predict.leaves(mod76, cadTrain, type = "where")

cadTrain$cluster51 = clusters51
cadTrain$cluster53 = clusters53
cadTrain$cluster55 = clusters55
cadTrain$cluster58 = clusters58
cadTrain$cluster62 = clusters62
cadTrain$cluster66 = clusters66
cadTrain$cluster70 = clusters70
cadTrain$cluster76 = clusters76

cadTrain$cluster51 = as.factor(cadTrain$cluster51)
cadTrain$cluster53 = as.factor(cadTrain$cluster53)
cadTrain$cluster55 = as.factor(cadTrain$cluster55)
cadTrain$cluster58 = as.factor(cadTrain$cluster58)
cadTrain$cluster62 = as.factor(cadTrain$cluster62)
cadTrain$cluster66 = as.factor(cadTrain$cluster66)
cadTrain$cluster70 = as.factor(cadTrain$cluster70)
cadTrain$cluster76 = as.factor(cadTrain$cluster76)

Clusters51 = cadTrain %>% 
  group_by(campaign, cluster51) %>% 
  summarise(count = length(cluster51),
            clicks = sum(click == 1),
            conversions = sum(conversion == 1))

Clusters51$combined <- paste(Clusters51$count, ",", Clusters51$clicks, ",", Clusters51$conversions)
Clusters512 = Clusters51
Clusters51 = Clusters51[, -c(3:5)]
Cluster51table = cast(Clusters51, cluster51 ~ campaign)

Clusters76 = cadTrain %>% 
  group_by(campaign, cluster76) %>% 
  summarise(count = length(cluster76),
            clicks = sum(click == 1),
            conversions = sum(conversion == 1))

#retrain

mod51 = rpart(click ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + cat9,
                 data = cadTrain, method="class", 
                 parms=list(split = 'gini'),
                 minbucket = 5, cp = 0.000003351605745)

mod55 = rpart(click ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + cat9,
                 data = cadTrain, method="class", 
                 parms=list(split = 'gini'),
                 minbucket = 5, cp = 0.000005336700392)

mod58 = rpart(click ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + cat9,
                 data = cadTrain, method="class", 
                 parms=list(split = 'gini'),
                 minbucket = 5, cp = 0.000007564638483)

mod62 = rpart(click ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + cat9,
                 data = cadTrain, method="class", 
                 parms=list(split = 'gini'),
                 minbucket = 5, cp = 0.00001204503519)

mod53 = rpart(click ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + cat9,
                 data = cadTrain, method="class", 
                 parms=list(split = 'gini'),
                 minbucket = 5, cp = 0.000004229245287)

mod66 = rpart(click ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + cat9,
                 data = cadTrain, method="class", 
                 parms=list(split = 'gini'),
                 minbucket = 5, cp = 0.00001917910793)

mod70 = rpart(click ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + cat9,
                 data = cadTrain, method="class", 
                 parms=list(split = 'gini'),
                 minbucket = 5, cp = 0.00003053857254)

mod76 = rpart(click ~ cat1 + cat2 + cat3 + cat4 + cat5 + cat6 + cat7 + cat8 + cat9,
              data = cadTrain, method="class", 
              parms=list(split = 'gini'),
              minbucket = 5, cp = 0.00006135910421)

Clusters53 = cadTrain %>% 
  group_by(campaign, cluster53) %>% 
  summarise(count = length(cluster53),
            clicks = sum(click == 1),
            conversions = sum(conversion == 1))

Clusters55 = cadTrain %>% 
  group_by(campaign, cluster55) %>% 
  summarise(count = length(cluster55),
            clicks = sum(click == 1),
            conversions = sum(conversion == 1))

Clusters58 = cadTrain %>% 
  group_by(campaign, cluster58) %>% 
  summarise(count = length(cluster58),
            clicks = sum(click == 1),
            conversions = sum(conversion == 1))

Clusters62 = cadTrain %>% 
  group_by(campaign, cluster62) %>% 
  summarise(count = length(cluster62),
            clicks = sum(click == 1),
            conversions = sum(conversion == 1))

Clusters66 = cadTrain %>% 
  group_by(campaign, cluster66) %>% 
  summarise(count = length(cluster66),
            clicks = sum(click == 1),
            conversions = sum(conversion == 1))

Clusters70 = cadTrain %>% 
  group_by(campaign, cluster70) %>% 
  summarise(count = length(cluster70),
            clicks = sum(click == 1),
            conversions = sum(conversion == 1))

Clusters76 = cadTrain %>% 
  group_by(campaign, cluster76) %>% 
  summarise(count = length(cluster76),
            clicks = sum(click == 1),
            conversions = sum(conversion == 1))

Clusters53$combined <- paste(Clusters53$count, ",", Clusters53$clicks, ",", Clusters53$conversions)
Clusters53 = Clusters53[, -c(3:5)]
Cluster53table = cast(Clusters53, cluster53 ~ campaign)

Clusters55$combined <- paste(Clusters55$count, ",", Clusters55$clicks, ",", Clusters55$conversions)
Clusters55 = Clusters55[, -c(3:5)]
Cluster55table = cast(Clusters55, cluster55 ~ campaign)

Clusters58$combined <- paste(Clusters58$count, ",", Clusters58$clicks, ",", Clusters58$conversions)
Clusters58 = Clusters58[, -c(3:5)]
Cluster58table = cast(Clusters58, cluster58 ~ campaign)

Clusters62$combined <- paste(Clusters62$count, ",", Clusters62$clicks, ",", Clusters62$conversions)
Clusters62 = Clusters62[, -c(3:5)]
Cluster62table = cast(Clusters62, cluster62 ~ campaign)

Clusters66$combined <- paste(Clusters66$count, ",", Clusters66$clicks, ",", Clusters66$conversions)
Clusters66 = Clusters66[, -c(3:5)]
Cluster66table = cast(Clusters66, cluster66 ~ campaign)

Clusters70$combined <- paste(Clusters70$count, ",", Clusters70$clicks, ",", Clusters70$conversions)
Clusters70 = Clusters70[, -c(3:5)]
Cluster70table = cast(Clusters70, cluster70 ~ campaign)

Clusters76$combined <- paste(Clusters76$count, ",", Clusters76$clicks, ",", Clusters76$conversions)
Clusters76 = Clusters76[, -c(3:5)]
Cluster76table = cast(Clusters76, cluster76 ~ campaign)

write.csv(Cluster51table, file = "train51.csv", na="")
write.csv(Cluster53table, file = "train53.csv", na="")
write.csv(Cluster55table, file = "train55.csv", na="")
write.csv(Cluster58table, file = "train58.csv", na="")
write.csv(Cluster62table, file = "train62.csv", na="")
write.csv(Cluster66table, file = "train66.csv", na="")
write.csv(Cluster70table, file = "train70.csv", na="")
write.csv(Cluster76table, file = "train76.csv", na="")

#test set

clusters51Test = rpart.predict.leaves(mod51, cadTest, type = "where")
clusters53Test = rpart.predict.leaves(mod53, cadTest, type = "where")
clusters55Test = rpart.predict.leaves(mod55, cadTest, type = "where")
clusters58Test = rpart.predict.leaves(mod58, cadTest, type = "where")
clusters62Test = rpart.predict.leaves(mod62, cadTest, type = "where")
clusters66Test = rpart.predict.leaves(mod66, cadTest, type = "where")
clusters70Test = rpart.predict.leaves(mod70, cadTest, type = "where")
clusters76Test = rpart.predict.leaves(mod76, cadTest, type = "where")

cadTest$cluster51 = clusters51Test
cadTest$cluster53 = clusters53Test
cadTest$cluster55 = clusters55Test
cadTest$cluster58 = clusters58Test
cadTest$cluster62 = clusters62Test
cadTest$cluster66 = clusters66Test
cadTest$cluster70 = clusters70Test
cadTest$cluster76 = clusters76Test

cadTest$cluster51 = as.factor(cadTest$cluster51)
cadTest$cluster53 = as.factor(cadTest$cluster53)
cadTest$cluster55 = as.factor(cadTest$cluster55)
cadTest$cluster58 = as.factor(cadTest$cluster58)
cadTest$cluster62 = as.factor(cadTest$cluster62)
cadTest$cluster66 = as.factor(cadTest$cluster66)
cadTest$cluster70 = as.factor(cadTest$cluster70)
cadTest$cluster76 = as.factor(cadTest$cluster76)

Clusters51Test = cadTest %>% 
  group_by(campaign, cluster51) %>% 
  summarise(count = length(cluster51),
            clicks = sum(click == 1),
            conversions = sum(conversion == 1))

Clusters53Test = cadTest %>% 
  group_by(campaign, cluster53) %>% 
  summarise(count = length(cluster53),
            clicks = sum(click == 1),
            conversions = sum(conversion == 1))

Clusters55Test = cadTest %>% 
  group_by(campaign, cluster55) %>% 
  summarise(count = length(cluster55),
            clicks = sum(click == 1),
            conversions = sum(conversion == 1))

Clusters58Test = cadTest %>% 
  group_by(campaign, cluster58) %>% 
  summarise(count = length(cluster58),
            clicks = sum(click == 1),
            conversions = sum(conversion == 1))

Clusters62Test = cadTest %>% 
  group_by(campaign, cluster62) %>% 
  summarise(count = length(cluster62),
            clicks = sum(click == 1),
            conversions = sum(conversion == 1))

Clusters66Test = cadTest %>% 
  group_by(campaign, cluster66) %>% 
  summarise(count = length(cluster66),
            clicks = sum(click == 1),
            conversions = sum(conversion == 1))

Clusters70Test = cadTest %>% 
  group_by(campaign, cluster70) %>% 
  summarise(count = length(cluster70),
            clicks = sum(click == 1),
            conversions = sum(conversion == 1))

Clusters76Test = cadTest %>% 
  group_by(campaign, cluster76) %>% 
  summarise(count = length(cluster76),
            clicks = sum(click == 1),
            conversions = sum(conversion == 1))

Clusters51Test$combined <- paste(Clusters51Test$count, ",", Clusters51Test$clicks, ",", Clusters51Test$conversions)
Clusters51Test = Clusters51Test[, -c(3:5)]
Cluster51tableTest = cast(Clusters51Test, cluster51 ~ campaign)

Clusters53Test$combined <- paste(Clusters53Test$count, ",", Clusters53Test$clicks, ",", Clusters53Test$conversions)
Clusters53Test = Clusters53Test[, -c(3:5)]
Cluster53tableTest = cast(Clusters53Test, cluster53 ~ campaign)

Clusters55Test$combined <- paste(Clusters55Test$count, ",", Clusters55Test$clicks, ",", Clusters55Test$conversions)
Clusters55Test = Clusters55Test[, -c(3:5)]
Cluster55tableTest = cast(Clusters55Test, cluster55 ~ campaign)

Clusters58Test$combined <- paste(Clusters58Test$count, ",", Clusters58Test$clicks, ",", Clusters58Test$conversions)
Clusters58Test = Clusters58Test[, -c(3:5)]
Cluster58tableTest = cast(Clusters58Test, cluster58 ~ campaign)

Clusters62Test$combined <- paste(Clusters62Test$count, ",", Clusters62Test$clicks, ",", Clusters62Test$conversions)
Clusters62Test = Clusters62Test[, -c(3:5)]
Cluster62tableTest = cast(Clusters62Test, cluster62 ~ campaign)

Clusters66Test$combined <- paste(Clusters66Test$count, ",", Clusters66Test$clicks, ",", Clusters66Test$conversions)
Clusters66Test = Clusters66Test[, -c(3:5)]
Cluster66tableTest = cast(Clusters66Test, cluster66 ~ campaign)

Clusters70Test$combined <- paste(Clusters70Test$count, ",", Clusters70Test$clicks, ",", Clusters70Test$conversions)
Clusters70Test = Clusters70Test[, -c(3:5)]
Cluster70tableTest = cast(Clusters70Test, cluster70 ~ campaign)

Clusters76Test$combined <- paste(Clusters76Test$count, ",", Clusters76Test$clicks, ",", Clusters76Test$conversions)
Clusters76Test = Clusters76Test[, -c(3:5)]
Cluster76tableTest = cast(Clusters76Test, cluster76 ~ campaign)

write.csv(Cluster51tableTest, file = "test51.csv", na="")
write.csv(Cluster53tableTest, file = "test53.csv", na="")
write.csv(Cluster55tableTest, file = "test55.csv", na="")
write.csv(Cluster58tableTest, file = "test58.csv", na="")
write.csv(Cluster62tableTest, file = "test62.csv", na="")
write.csv(Cluster66tableTest, file = "test66.csv", na="")
write.csv(Cluster70tableTest, file = "test70.csv", na="")
write.csv(Cluster76tableTest, file = "test76.csv", na="")
