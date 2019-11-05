
#install.packages("MASS")
#install.packages("randomForest")
#install.packages("ggplot2")
#install.packages("tibble")
#install.packages("dplyr")
#install.packages("ipred")
set.seed(125)
library(MASS)
library(randomForest)
library(ggplot2)
library(tibble)
library(dplyr)
library(ipred)

data("Boston")
index <- sample(nrow(Boston),nrow(Boston)*0.70) # separar train/test
boston.train <- Boston[index,]
boston.test <- Boston[-index,]
rm(Boston)

head(Boston)

#install.packages("ipred")
library(ipred)

boston.bag<- bagging(medv~., data = boston.train, nbagg=100)
boston.bag.pred<- predict(boston.bag, newdata = boston.test)
mean((boston.test$medv-boston.bag.pred)^2)
rm(boston.bag,boston.bag.pred)

#install.packages("rpart")
library(rpart)


boston.tree<- rpart(medv~., data = boston.train)
boston.tree.pred<- predict(boston.tree, newdata = boston.test)
mean((boston.test$medv-boston.tree.pred)^2)
rm(boston.tree,boston.tree.pred)

ntree<- c(1, 3, 5, seq(10, 200, 10))
MSE.test<- rep(0, length(ntree))
for(i in 1:length(ntree)){
  boston.bag1<- bagging(medv~., data = boston.train, nbagg=ntree[i])
  boston.bag.pred1<- predict(boston.bag1, newdata = boston.test)
  MSE.test[i]<- mean((boston.test$medv-boston.bag.pred1)^2)
}
plot(ntree, MSE.test, type = 'l', col=2, lwd=2)
rm(boston.bag1,boston.bag.pred1)

boston.bag.oob<- bagging(medv~., data = boston.train, coob=T, nbagg=100)
boston.bag.oob # coob=T significa que sea TRUE el uso de OOB
rm(boston.bag.oob)

library(randomForest)
boston.rf<- randomForest(medv~., data = boston.train, importance=TRUE)
boston.rf

tabla<- rownames_to_column(data.frame(boston.rf$importance),"Variable")
tabla

tabla %>% ggplot(aes(y=X.IncMSE,x=Variable))+geom_bar(stat="identity")+
  coord_flip()+theme_classic()+ 
  geom_text(aes(x=Variable,y=X.IncMSE,label=X.IncMSE),hjust=0)

plot(boston.rf$mse, type='l', col=2, lwd=2, xlab = "ntree", ylab = "OOB Error")

boston.rf.pred<- predict(boston.rf, boston.test)
mean((boston.test$medv-boston.rf.pred)^2)
rm(boston.rf)

oob.err<- rep(0, 13)
test.err<- rep(0, 13)
for(i in 1:13){
  fit<- randomForest(medv~., data = boston.train, mtry=i)
  oob.err[i]<- fit$mse[500]
  test.err[i]<- mean((boston.test$medv-predict(fit, boston.test))^2)
  cat(i, " ")
}
matplot(cbind(test.err, oob.err), pch=15, col = c("red", "blue"), type = "b", ylab = "MSE", xlab = "mtry")+
  legend("topright", legend = c("test Error", "OOB Error"), pch = 15, col = c("red", "blue"))

install.packages("gbm")
library(gbm)

boston.boost<- gbm(medv~., data = boston.train, distribution = "gaussian",
                   n.trees = 10000, shrinkage = 0.01, interaction.depth = 8)
summary(boston.boost)

boston.boost.pred.test<- predict(boston.boost, boston.test, n.trees = 10000)
mean((boston.test$medv-boston.boost.pred.test)^2)

ntree<- seq(100, 10000, 100)
predmat<- predict(boston.boost, newdata = boston.test, n.trees = ntree)
err<- apply((predmat-boston.test$medv)^2, 2, mean)
plot(ntree, err, type = 'l', col=2, lwd=2, xlab = "n.trees", ylab = "Test MSE")
abline(h=min(test.err), lty=2)

#install.packages("mlbench")
#install.packages("caret")
#install.packages("e1071")
library(mlbench)
library(caret)
library(e1071)
set.seed(1)



data(Sonar)
Base <- Sonar
Variables <- Base[,1:60]
Clase <- Base[,61]

inTraining <- createDataPartition(Base$Class, p = .70, list =FALSE)

#Training corresponde al 70% de la base train
training <- Base[inTraining,]

#Testing corresponde al restante 30% de la base train
testing  <- Base[-inTraining,]


head(Sonar)

control_svm <- trainControl(method='cv', 
                        number=10,
                            
                        verboseIter = TRUE,
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE)
# Entrenamiento de la base training utilizando el metodo de support vector machine con AUTOMATIC GRID
model_SVM_AG <- train(
  Class ~ ., training,
  method = "svmPoly",
  trControl = control_svm,
  tuneLength = 6,
  metric ="ROC"
  )





plot(model_SVM_AG)

print(model_SVM_AG)

model_SVM <- train(
  Class ~ ., training,
  method = "svmPoly",
    metric ="ROC",
  trControl = trainControl(
    method = "cv", number = 10,
    verboseIter = TRUE,
      summaryFunction = twoClassSummary,
                        classProbs = TRUE
  )
)



#Aplicar el modelo en la base testing
print("MODELO NORMAL")
SVMprediction <-predict(model_SVM, testing)
cmSVM <-confusionMatrix(SVMprediction,testing$Class)
print(cmSVM)
print("MODELO CON AUTOMATIC GRID")
SVM_AG_predict <-predict(model_SVM_AG, testing)
cmSVM_AG1 <-confusionMatrix(SVM_AG_predict,testing$Class)
print(cmSVM_AG1)

Control_DT_c50 <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 3,verboseIter = TRUE,
                           summaryFunction = twoClassSummary,
                           classProbs = TRUE)

#Modelo DT Original
model_DT <- train(
  Class ~ ., training,
  method = "C5.0",
  metric = "ROC",
  trControl = trainControl(
    method = "cv", number = 10,
    verboseIter = TRUE,
    summaryFunction = twoClassSummary,
    classProbs = TRUE))

#Modelo DT con GridSearch
model_DT_1 <- train(
  Class ~ ., training,
  method = "C5.0",
  trControl = Control_DT_c50, 
  tuneLength = 10,
  metric ="ROC")
    

print("MODELO ORIGINAL")
DTprediction <-predict(model_DT, testing)
cmDT <-confusionMatrix(DTprediction,testing$Class)
print(cmDT)
print("MODELO CON AUTOMATIC GRID")
DTprediction_GD <-predict(model_DT_1, testing)
cmDT_GD <-confusionMatrix(DTprediction_GD,testing$Class)
print(cmDT_GD)

#primero definimos los fold y las repeticiones del crossvalidation
Control_DT_c50 <- trainControl(method = "repeatedcv",
                               number = 10,
                               repeats = 3,verboseIter = FALSE,
                              summaryFunction = twoClassSummary,
                               classProbs = TRUE)
#Segundo definimos la grilla mediante vectores

grid_DT <- expand.grid( .winnow = c(TRUE,FALSE), .trials=c(7,15,17,18,20), .model="tree" )

model_DT_2 <- train(
  Class ~ ., training,
  tuneGrid=grid_DT,
  method = "C5.0",
    metric ="ROC",
  trControl = Control_DT_c50)


plot(model_DT_2)
print(model_DT_2)
#Aplicar el modelo en la base testing


print("MODELO ORIGINAL")
print(cmDT)
print("MODELO MANUAL GRID")
DTprediction_MG <-predict(model_DT_2, testing)
cmDT_MG <-confusionMatrix(DTprediction_MG,testing$Class)
print(cmDT_MG)



#install.packages("randomForest")
library(randomForest)

#Definimos la configuración de entrenamiento como un cross validation 10 fold con 3 repeticiones
control_rf <- trainControl(method="repeatedcv", number=10, repeats=3, summaryFunction = twoClassSummary,
                               classProbs = TRUE)
# definimos los valores para mtry seran del 1 al 20 
grid_rf <- expand.grid(.mtry=c(1:20))
rf <- train(Class~., data=training, method="rf", metric="ROC", tuneGrid=grid_rf, trControl=control_rf)
print(rf)
plot(rf)


#Definimos la configuración de entrenamiento como un cross validation 10 fold sin repeticiones
control_rf_2 <- trainControl(method="cv", number=10, summaryFunction = twoClassSummary,
                               classProbs = TRUE)
# definimos los valores para mtry seran del 1 al 20 
rf_grid2 <- expand.grid(.mtry=c(1:20))

#creamos una lista vacia para ir guardando los distintos modelos
rf_list <- list()

# mediante un loop vamos probando los distintos valores de ntree. 
## estos van desde el 100 al 500 cada 50. Es decir 100 - 150 - 200 - 250 .......- 400 - 450 - 500
for (i in seq(200, 600, by = 50)) {
  set.seed(1)
  rf4 <- train(Class ~., 
               data = training, 
               method = "rf", 
               metric = "ROC", 
               tuneGrid = rf_grid2, 
               ntree = i, 
               trControl = control_rf_2)
  key <- toString(i)
  rf_list[[key]] <- rf4
  
}

bosques <- resamples(rf_list)
summary(bosques)

dotplot(results)

rf_list$`500`$bestTune

rf_list$`500`$results

rf_list$`500` %>%
  plot()


