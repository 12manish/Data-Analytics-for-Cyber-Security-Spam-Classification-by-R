packageslist <- list("readr", "caret", "MASS", "klaR", "randomForest")
library(dplyr)
#install.packages("klaR")
library(klaR)
library(ggplot2)
library(DataExplorer)
load_packages <- lapply(packageslist, require, character.only = T)



setwd("C:\\Users\\SPEED TAIL\\Desktop\\CyberSecurity\\Data analytics Cyber Security\\dataset")
#-------Importing the data---------
train<-read.table("training_data.txt",sep = ",")
test1<-read.table("testing_data1.txt",sep = ",")
test2<-read.table("testing_data2.txt",sep = ",")
##Now giving the header
colnames(train)<-c("account_age","no_followers","no_folowing","no_userfavirate","No_lists","no_tweets","no_retweets","no_tweetsfaviorate","no_hashtag","no_usermention","no_urls","no_char","no_digits","spam")
colnames(test1)<-c("account_age","no_followers","no_folowing","no_userfavirate","No_lists","no_tweets","no_retweets","no_tweetsfaviorate","no_hashtag","no_usermention","no_urls","no_char","no_digits","spam")
colnames(test2)<-c("account_age","no_followers","no_folowing","no_userfavirate","No_lists","no_tweets","no_retweets","no_tweetsfaviorate","no_hashtag","no_usermention","no_urls","no_char","no_digits","spam")
##Now check renamed colnames
colnames(train)
dim(train)
dim(test1)
dim(test2)
colnames(test1)
colnames(test2)
View(train)
View(test1)
View(test2)
plot_missing(test2)
plot_missing(train)
plot_missing(test1)

# Data Exploration

dim(train)
dim(test1)
dim(test2)

str(train)
str(test1)
str(test2)

summary(train)
summary(test1)
summary(test2)
### top 5 lower and top 5 higest and unique, mean,
#all percentile show,n, missing also
#install.packages("Hmisc")
library(Hmisc)
describe(train)
describe(test1)
describe(test2)
# for check data frequency of missing values
#install.packages("questionr")
library(questionr)
freq.na(train)
freq.na(test1)
freq.na(test2)
# Checking the frequency Distribution of the dependent variable spammer or non-spammer

table(train$spam)
table(train$spam)/nrow(train)
table(test1$spam)/nrow(test1)
table(test2$spam)/nrow(test2)

# change label of last column for convenience
colnames(train)[14] = "Spam"   
colnames(test1)[14] = "Spam" 
colnames(test2)[14] = "Spam" 

## Meke Spam dummy
train$Spam <- ifelse(train$Spam == "spammer",1,0)
test1$Spam <- ifelse(test1$Spam == "spammer",1,0)
test2$Spam <- ifelse(test2$Spam == "spammer",1,0)
train$Spam = as.factor(train$Spam)
test1$Spam = as.factor(test1$Spam)
test2$Spam = as.factor(test2$Spam)

colnames(train)
#Data Visualization
#Visual 1
ggplot(data=train, aes(x=account_age, y=no_urls, color=Spam)) +
  #geom_line(alpha=1)+
  geom_smooth(se=F,aes(color = Spam), size = 1)+
ggtitle("Distribution for Spam")

ggplot(data=train, aes(x=no_followers, y=no_folowing, color=Spam)) +
  #geom_line(alpha=1)+
  geom_smooth(se=F,aes(color = Spam), size = 1)+
ggtitle("Distribution for Spam")

ggplot(data=train, aes(x=no_tweets, y=no_retweets, color=Spam)) +
  #geom_line(alpha=1)+
  geom_smooth(se=F,aes(color = Spam), size = 1)+
ggtitle("Distribution for Spam")

ggplot(data=train, aes(x=no_followers, y=no_tweetsfaviorate, color=Spam)) +
  #geom_line(alpha=1)+
  geom_smooth(se=F,aes(color = Spam), size = 1)+
ggtitle("Distribution for Spam")

ggplot(data=train, aes(x=no_followers, y=no_usermention, color=Spam)) +
  #geom_line(alpha=1)+
  geom_smooth(se=F,aes(color = Spam), size = 1)+
ggtitle("Distribution  for Spam")


ggplot(train, aes(account_age, colour = Spam)) +
  geom_freqpoly(binwidth = 1) + labs(title="Age Distribution for Spam")
ggplot(train, aes(no_followers, colour = Spam)) +
  geom_freqpoly(binwidth = 10) + labs(title="no_followers Distribution for Spam")
ggplot(train, aes(no_followers, colour = Spam)) +
  geom_freqpoly(binwidth = 10) + labs(title="no_followers Distribution by Spam")

ggplot(train, aes(no_userfavirate, colour = Spam)) +
  geom_freqpoly(binwidth = 10) + labs(title="no_userfavirate Distribution by Spam")

ggplot(train, aes(no_followers, colour = Spam)) +
  geom_freqpoly(binwidth = 10) + labs(title="no_followers Distribution by Spam")
ggplot(train, aes(no_urls, colour = Spam)) +
  geom_freqpoly(binwidth = 1) + labs(title="no_followers Distribution by Spam")
#visual 2
c <- ggplot(train, aes(x=no_urls, fill=Spam, color=Spam)) +
  geom_histogram(binwidth = 1) + labs(title="no_urls Distribution by Spam")
c + theme_bw()

#visual 3
P <- ggplot(train, aes(x=no_digits, fill=Spam, color=Spam)) +
  geom_histogram(binwidth = 1) + labs(title="no digit Distribution by Spam")
P + theme_bw()
P <- ggplot(train, aes(x=no_char, fill=Spam, color=Spam)) +
  geom_histogram(binwidth = 1) + labs(title="no_char Distribution by Spam")
P + theme_bw()
#visual 4
ggplot(train, aes(no_hashtag, colour = Spam)) +
  geom_freqpoly(binwidth = 1) + labs(title="no_hashtag Distribution by Spam")



#####Machine learning model start
library(randomForest) # randomForest method
library(MASS) # linear discrimant analysis
library(nnet) # neural network
library(caret)

library(kernlab)
#install.packages("doMC", repos="http://R-Forge.R-project.org")
library(doMC)
#Cross validation
##We will be using K- fold cross validation, with number of folds (k) set at 10.
#tcontrol <- trainControl(method = "cv", number = 10)

#set up the training control
tr_ctrl = trainControl(method = "cv",number = 10)
#train the models
forest_train = train(Spam ~ ., 
                     data=train, 
                     method="rf",
                     trControl=tr_ctrl)
forest_train$results

fit_rf <- randomForest(Spam ~., data = train)
summary(fit_rf)
(VI_F=importance(fit_rf))
varImp(fit_rf)
varImpPlot(fit_rf,type=2,sort = T, main = "variable Importance")
p = predict(fit_rf,newdata=test2,type="prob")

nnet_train = train(Spam ~ ., 
                   data=train, 
                   method="nnet",
                   trControl=tr_ctrl)
nnet_train$results
# Fitting SVM to the Training set 
#install.packages('e1071') 
library(e1071) 



#Classification. Predictive Model. RPart (Recursive Partitioning and Regression Trees) Algorithm
library(rpart)
pc <- proc.time()
model.rpart <- rpart(Spam ~ ., method = "class", data = train)
proc.time() - pc
printcp(model.rpart)

plot(model.rpart, uniform = TRUE, main = "Classification (RPART). Classification Tree for SPAM")
text(model.rpart, all = TRUE, cex = 0.75)
#install.packages("rattle")
#library(rattle)
install.packages("FactoMineR")
library(rpart)
library(rpart.plot)
library(FactoMineR)
install.packages("maptree")
library(maptree)
draw.tree(model.rpart, cex = 0.5, nodeinfo = TRUE, col = gray(0:8/8))
##Confusion Matrix (RPart)
prediction.rpart <- predict(model.rpart, newdata = test2, type = "class")
table(`Actual Class` = test2$Spam, `Predicted Class` = prediction.rpart)
error.rate.rpart <- sum(test2$Spam != prediction.rpart)/nrow(test2)
print(paste0("Accuary (Precision): ", 1 - error.rate.rpart))
caret::confusionMatrix(prediction.rpart, test2$Spam)

#Classification. Predictive Model. SVM (Support Vector Machine) Algorithm
pc <- proc.time()
model.svm <- svm(Spam ~ ., method = "class", data = train)
proc.time() - pc
summary(model.svm)
##Confusion Matrix (SVM)
prediction.svm <- predict(model.svm, newdata = test2, type = "class")
table(`Actual Class` = test2$Spam, `Predicted Class` = prediction.svm)
error.rate.svm <- sum(test2$Spam != prediction.svm)/nrow(test2)
print(paste0("Accuary (Precision): ", 1 - error.rate.svm))
caret::confusionMatrix(prediction.svm, test2$Spam)



pred_forest = predict(forest_train, test2)

pred_nnet = predict(nnet_train, test2)
##random forest
caret::confusionMatrix(pred_forest, test2$Spam)


##neural network
caret::confusionMatrix(pred_nnet, test2$Spam)


###Classification. Predictive Model. RPart (Recursive Partitioning and Regression Trees)Algorithm
library(rpart)
library(plogr)
install.packages("maptree")
library(maptree)
pc <- proc.time()
model.rpart <- rpart(Spam ~ ., method = "class", data = train)
proc.time() - pc

printcp(model.rpart)
plot(model.rpart, uniform = TRUE, main = "Classification (RPART). Classification Tree for SPAM")
text(model.rpart, all = TRUE, cex = 0.75)

draw.tree(model.rpart, cex = 0.52, nodeinfo = TRUE, col = gray(0:8/8))
##Confusion Matrix (RPart)
prediction.rpart <- predict(model.rpart, newdata = test2, type = "class")
table(`Actual Class` = test2$Spam, `Predicted Class` = prediction.rpart)
error.rate.rpart <- sum(test2$Spam != prediction.rpart)/nrow(test2)
print(paste0("Accuary (Precision): ", 1 - error.rate.rpart))
caret::confusionMatrix(prediction.rpart, test2$Spam)

library(e1071) ##Naive Bias
control <- trainControl(method="repeatedcv", number=10, repeats=3)
system.time( classifier_nb <- naiveBayes(train, train$Spam, laplace = 1,
                                         trControl = control,tuneLength = 7) )
##Making Predictions and evaluating the Naive Bayes Classifier.
nb_pred = predict(classifier_nb, type = 'class', newdata = test2)
caret::confusionMatrix(nb_pred, test2$Spam)






##############MODELLLLLLLLL#############



tcontrol <- trainControl(method = "repeatedcv",number = 14,repeats = 2)
# KNN
modelKNN <- train(Spam ~ ., data = train, method = "knn", preProcess = c("center", 
                                                                                 "scale"), trControl = tcontrol)  # data is normalised using Preprocess
# Naive Bayes
library(e1071)
modelNB <- train(Spam ~ ., data = train, method = "nb", trControl = tcontrol)
# Random Forest
modelRF <- train(Spam ~ ., data = train, method = "rf", 
                 importance = T, trControl = tcontrol)
##modelda <- lda(Spam~., data = train,method = "mle",trControl = tcontrol)

# Logisitic Regression
modelLG <- train(Spam ~ ., data = train, method = "glm", family = binomial, 
                 trControl = tcontrol)

library(nnet) # neural network
modelNet = train(Spam ~ ., 
                   data=train, 
                   method="nnet",
                   trControl=tcontrol)


##Predict on validation set(test2)
##We will make use of the train models and make predicitions on validation set.

# KNN
pKNN <- predict(modelKNN, test2)
# Naive Bayes
pNB <- predict(modelNB, test2)
# Random Forest
pRF <- predict(modelRF, test2)
# Logistic Regression
pLG <- predict(modelLG, test2)

#Neural network
pNET<- predict(modelNet, test2)


##Confusion matrix
##Predictions from the validation(test2) set can be compared to the actual outcomes 
##to create a confusion matrix for each model.

# KNN
cmKNN <- confusionMatrix(test2$Spam, pKNN)
# Naive Bayes
cmNB <- confusionMatrix(test2$Spam, pNB)
# Random Forest
cmRF <- confusionMatrix(test2$Spam, pRF)
# Logisitic Regression
cmLG <- confusionMatrix(test2$Spam, pLG)
#neural network
pnet<- confusionMatrix(test2$Spam, pNET)

##Lets put all of this together in a table.
ModelType <- c("K nearest neighbor", "Naive Bayes", "Random forest", "Logistic regression","Neural Network")  # vector containing names of models

# Training classification accuracy
TrainAccuracy <- c(max(modelKNN$results$Accuracy), max(modelNB$results$Accuracy), 
                   max(modelRF$results$Accuracy),max(modelNet$results$Accuracy),
                   max(modelLG$results$Accuracy))

# Training misclassification error
Train_missclass_Error <- 1 - TrainAccuracy

# validation classification accuracy
ValidationAccuracy <- c(cmKNN$overall[1], cmNB$overall[1], cmRF$overall[1], pnet$overall[1],
                        cmLG$overall[1])

# Validation misclassification error or out-of-sample-error
Validation_missclass_Error <- 1 - ValidationAccuracy

metrics <- data.frame(ModelType, TrainAccuracy, Train_missclass_Error, ValidationAccuracy, 
                      Validation_missclass_Error)  # data frame with above metrics

knitr::kable(metrics, digits = 5)  # print table using kable() from knitr package

#Predicting Test Values
pTestingRF <- predict(modelRF, test1)
pTestingRF

pTestingKNN <- predict(modelKNN, test1)
pTestingKNN






library(neuralnet)
install.packages("NeuralNetTools")
library(NeuralNetTools)
library(plyr)
Startups <- as.data.frame(train)

# Exploratory data Analysis :
colnames(train)
plot(train$account_age, train$Spam)
# Correlation coefficient - Strength & Direction of correlation
cor(Startups[,-14])
summary(Startups) # Confirms on the different scale and demands normalizing the data.
##Apply Normalization technique to the whole dataset :
  
Startups<-scale(Startups[,-14])
# Creating a neural network model on training data

startups_model <- neuralnet(Spam ~ ., data = train)
str(startups_model)
plot(startups_model, rep = "best")
summary(startups_model)
par(mar = numeric(4), family = 'serif')
plotnet(startups_model, alpha = 0.6)
# Evaluating model performance

set.seed(12323)
model_results <- compute(startups_model,test1)
predicted_Spam <- model_results$net.result

test1$Spam<-as.numeric(test1$Spam)
# Predicted profit Vs Actual profit of test data.
cor(predicted_Spam,test1$Spam)


# SSE(Error) has reduced and training steps had been increased 
#as the number of neurons  under hidden layer are increased