##The goal of this project is to test three different supervised machine learning algorithms 
##that could potentially be used as a spam filter. Essentially we want to see which algorithm 
##has the best chance of predicting a spam email given certain criteria.



library(dplyr)
library(GGally)
library(corrplot)
library(caret)
library(psych)
library(rpart)
library(randomForest)
library(nnet)
library(e1071)
library(naivebayes)
setwd("C:\\Users\\SPEED TAIL\\Desktop\\Network cyber")
#-------Importing the data---------
train<-read.table("training_data.txt",sep = ",")
test1<-read.table("testing_data1.txt",sep = ",")
test2<-read.table("testing_data2.txt",sep = ",")
colnames(train)<-c("account_age","no_followers","no_folowing","no_userfavirate","No_list","no_tewwts","no_retweets","no_tweetsfaviorate","no_hashtag","no_usermention","no_urls","no_char","no_digits","spam")
colnames(test1)<-c("account_age","no_followers","no_folowing","no_userfavirate","No_list","no_tewwts","no_retweets","no_tweetsfaviorate","no_hashtag","no_usermention","no_urls","no_char","no_digits","spam")
colnames(test2)<-c("account_age","no_followers","no_folowing","no_userfavirate","No_list","no_tewwts","no_retweets","no_tweetsfaviorate","no_hashtag","no_usermention","no_urls","no_char","no_digits","spam")

##read.delim(file.choose())
colnames(train)
colnames(test1)
colnames(test2)
View(train)
View(test1)
View(test2)

install.packages("fastDummies")
library(fastDummies)
trainf <- fastDummies::dummy_cols(train, select_columns = "spam")
testf1 <- fastDummies::dummy_cols(test1, select_columns = "spam")
testf2 <- fastDummies::dummy_cols(test2, select_columns = "spam")

# Data Exploration

dim(trainf)
dim(testf1)
dim(testf2)

str(trainf)
str(testf1)
str(testf2)

summary(trainf)
summary(testf1)
summary(testf2)

### top 5 lower and top 5 higest and unique, mean,
#all percentile show,n, missing also
install.packages("Hmisc")
library(Hmisc)
describe(trainf)
describe(testf1)
describe(testf2)



#Checking for missing values in all the columns

colSums(is.na(trainf))
# for check data frequency of na
install.packages("questionr")
library(questionr)
freq.na(trainf)
freq.na(testf1)
freq.na(testf2)

# Checking the frequency Distribution of the target variable spammer or non-spammer

table(trainf$spam)
table(trainf$spam)/nrow(trainf)
table(testf1$spam)/nrow(testf1)
## in real world, there are only around 5% spam tweets in Twitter,
##which indicates that testing Dataset 2 simulates the real-world scenario
table(testf2$spam)/nrow(testf2)    ##non-spanner 0.95 and spanner 0.05


##Remove the spam categoriacl column from train and test data (have dummy variables)
names(trainf)
#trainf<-trainf[,-14]
#testf1<-testf1[,-14]
#testf2<-testf2[,-14]

##Needed <- c("tm", "SnowballCC", "RColorBrewer", "ggplot2", "wordcloud", "biclust", "cluster", "igraph", "fpc")

# Reading Most Frequent Terms
##sort(colSums(trainf))

# Logistic Regression Model
## we are creating a logistic regression model called as spamLog, 
## for logistic regression we use function glm with family binomial. 
## we are using the all the variable as independent variable for training our model, for detection.



#####Prep Data for Training
##The spambase dataset contains a set of word frequencies occuring in emails. 
##Each email was labelled spam or not spam, denoted as 0 or 1. 
##The spam column(V14) is turned into a factor since we are testing binary classification.


colnames(trainf)[14] = "Spam"   # change label of last column for convenience
colnames(testf1)[14] = "Spam" 
colnames(testf2)[14] = "Spam" 
## Meke spam dummy
trainf$Spam <- ifelse(trainf$Spam == "spammer",1,0)
testf1$Spam <- ifelse(testf1$Spam == "spammer",1,0)
testf2$Spam <- ifelse(testf2$Spam == "spammer",1,0)
trainf$Spam = as.factor(trainf$Spam)
testf1$Spam = as.factor(testf1$Spam)
testf2$Spam = as.factor(testf2$Spam)
View(trainf)
library(randomForest) # randomForest method

library(MASS) # linear discrimant analysis
library(nnet) # neural network

#Training the Models
##After some research I decided to test Random Forest, Linear Discriminant Analysis and 
###Neural Networks. These algorithms required the least tuning and are designed to work with 
##categorical data.

##These algorithms are masked from their respective packages and used with caret's 
##training function. The training control method used is repeated cross validation. 10 seperate 10 fold cross-overs are used to improve model accuracy.
#set up the training control
##The training control method used is repeated cross validation.
##10 seperate 10 fold cross-overs are used to improve model accuracy.
tr_ctrl = trainControl(method = "repeatedcv",number = 10,repeats = 10)
#train the models
forest_train = train(Spam ~ ., 
                     data=trainf, 
                     method="rf",
                     trControl=tr_ctrl)
summary(forest_train)

install.packages("randomForest")
library(randomForest)
fit_rf <- randomForest(spam ~., data = train)
summary(fit_rf)
(VI_F=importance(fit_rf))
varImp(fit_rf)
varImpPlot(fit_rf,type=2)
p = predict(fit_rf,newdata=testf2,type="prob")

library(e1071) ##Naive Bias
control <- trainControl(method="repeatedcv", number=10, repeats=3)
system.time( classifier_nb <- naiveBayes(trainf, trainf$Spam, laplace = 1,
                                         trControl = control,tuneLength = 7) )
##Making Predictions and evaluating the Naive Bayes Classifier.
nb_pred = predict(classifier_nb, type = 'class', newdata = testf2)

confusionMatrix(nb_pred,testf2$Spam)

#The Naive Bayes Classifier also performed very well on the training set by achieving 99.70% accuracy
##which means we have got 7 misclassifications out a possible 1209 observation. 
##While the model has a 99% sensitivity rate; the proportion of the positive class
##predicted as positive, it was able to achieve about 99% on specificity rate which is the 
##proportion of the negative class predicted accurately 


##Support Vector Machine
svm_classifier <- svm(Spam~., data=trainf)
svm_classifier
##Making Predictions and evaluating the Support Vector Machine Classifier
svm_pred = predict(svm_classifier,testf2)

confusionMatrix(svm_pred,testf2$Spam)
library(nnet) # neural network
nnet_train = train(Spam ~ ., 
                   data=trainf, 
                   method="nnet",
                   trControl=tr_ctrl)
### Different model######



##Results
###Now that the models are trained we test them against the testing set. 
##The confusionMatrix() displays the results and relevent statistics for each model.

pred_forest = predict(forest_train, testf2[,-14])
##pred_lda = predict(lda_train, testf1[,-14])
pred_nnet = predict(nnet_train, testf2[,-14],type="raw")
summary(pred_nnet)
confusionMatrix(pred_forest,testf2$Spam)
str(testf1)
str(testf2)
caret::confusionMatrix(pred_nnet, testf2$Spam)
##confusionMatrix(pred_nnet,testf1$spam)
##model evalution for Nnet model
##Diff methods------------------------------------------############ 
test_predictionsbyNnet = predict(nnet_train, testf1)
test_predictionsbyNnet

conf_matrix = confusionMatrix(test_predictionsbyNnet, testf1$Spam)
conf_matrix
caret::confusionMatrix(pred_nnet, testf1$Spam)


# Prediction & Confusion Matrix - test data
p1 <- predict(nnet_train, testf1, type="raw")
head(p1)
table(testf2$Spam)/nrow(testf2)

confusionMatrix(p1, testf1$Spam,positive = "1")



##Result#########################
##With this sample, Random Forest seemed to perform the best with 94% accuracy and a confidence interval that suggests up to 96% accuracy.
##LDA and Neural Networks performed slightly lower,but with fine tuning the results would most likely improve.


###Classification. Predictive Model. RPart (Recursive Partitioning and Regression Trees)Algorithm
library(rpart)
library(plogr)
install.packages("maptree")
library(maptree)
pc <- proc.time()
model.rpart <- rpart(Spam ~ ., method = "class", data = trainf)
proc.time() - pc

printcp(model.rpart)
plot(model.rpart, uniform = TRUE, main = "Classification (RPART). Classification Tree for SPAM")
text(model.rpart, all = TRUE, cex = 0.75)

draw.tree(model.rpart, cex = 0.52, nodeinfo = TRUE, col = gray(0:8/8))
##Confusion Matrix (RPart)
prediction.rpart <- predict(model.rpart, newdata = testf2, type = "class")
table(`Actual Class` = testf2$Spam, `Predicted Class` = prediction.rpart)
error.rate.rpart <- sum(testf2$Spam != prediction.rpart)/nrow(testf2)
print(paste0("Accuary (Precision): ", 1 - error.rate.rpart))


##Classification. Predictive Model. SVM (Support Vector Machine) Algorithm
pc <- proc.time()
model.svm <- svm(Spam ~ ., method = "class", data = trainf)
proc.time() - pc

summary(model.svm)
##Confusion Matrix (SVM)
prediction.svm <- predict(model.svm, newdata = testf2, type = "class")
table(`Actual Class` = testf2$Spam, `Predicted Class` = prediction.svm)
error.rate.svm <- sum(testf2$Spam != prediction.svm)/nrow(testf2)
print(paste0("Accuary (Precision): ", 1 - error.rate.svm))

##Classification. Predictive Model. Naive Bias Algorithm

library(e1071)

library(caTools)

install.packages("Ks")
library(Ks)

library(naivebayes)
#using KNN
library(class)

#build train and test data
#Applying K-NN model 
# Fitting SVM to the Training set 
install.packages('e1071') 
library(e1071) 

classifier = svm(formula = Spam ~ ., 
                 data = trainf, 
                 type = 'C-classification', 
                 kernel = 'linear') 

# Predicting the Test set results 
y_pred = predict(classifier, newdata = testf2[,-14]) 
# Making the Confusion Matrix 

conf_matrix = confusionMatrix(y_pred, testf2$Spam)
conf_matrix
caret::confusionMatrix(y_pred, testf2$Spam)
########Visualization#############

#Exploratory Data Analysis
colnames(trainf)
#Data Visualization
#Visual 1
ggplot(trainf, aes(account_age, colour = Spam)) +
  geom_freqpoly(binwidth = 1) + labs(title="Age Distribution by Spam")
ggplot(trainf, aes(no_followers, colour = Spam)) +
  geom_freqpoly(binwidth = 10) + labs(title="no_followers Distribution by Spam")
ggplot(trainf, aes(no_followers, colour = Spam)) +
  geom_freqpoly(binwidth = 10) + labs(title="no_followers Distribution by Spam")

ggplot(trainf, aes(no_userfavirate, colour = Spam)) +
  geom_freqpoly(binwidth = 10) + labs(title="no_userfavirate Distribution by Spam")

ggplot(trainf, aes(no_followers, colour = Spam)) +
  geom_freqpoly(binwidth = 10) + labs(title="no_followers Distribution by Spam")
ggplot(trainf, aes(no_urls, colour = Spam)) +
  geom_freqpoly(binwidth = 1) + labs(title="no_followers Distribution by Spam")
#visual 2
c <- ggplot(trainf, aes(x=no_urls, fill=Spam, color=Spam)) +
  geom_histogram(binwidth = 1) + labs(title="no_urls Distribution by Spam")
c + theme_bw()

#visual 3
P <- ggplot(trainf, aes(x=no_digits, fill=Spam, color=Spam)) +
  geom_histogram(binwidth = 1) + labs(title="no digit Distribution by Spam")
P + theme_bw()
P <- ggplot(trainf, aes(x=no_char, fill=Spam, color=Spam)) +
  geom_histogram(binwidth = 1) + labs(title="no_char Distribution by Spam")
P + theme_bw()
#visual 4
ggplot(trainf, aes(no_hashtag, colour = Spam)) +
  geom_freqpoly(binwidth = 1) + labs(title="no_hashtag Distribution by Spam")
#visual 5
ggpairs(trainf)


## Naive bias
library(e1071)
x_train = trainf[,-14]
y_train = trainf$Spam
x_test = testf2[,-14]
y_test = testf2$Spam
x <- cbind(x_train,y_train)
# Fitting model
fit <-naiveBayes(y_train ~ ., data = x)
summary(fit)
#Predict Output 
predicted= predict(fit,x_test)
#Confusion matrix for Naive Bias
caret::confusionMatrix(predicted, testf2$Spam)
##SVM
library(e1071)
S <- cbind(x_train,y_train)
# Fitting model
fit <-svm(y_train ~ ., data = S)
summary(fit)
#Predict Output 
predictedsvm= predict(fit,x_test)
#Confusion matrix for Naive Bias
caret::confusionMatrix(predictedsvm, testf2$Spam)

install.packages(c("knitr","RColorBrewe","gridBase","ElemStatLearn","foreign","tree","rpart","maptree","class","ROCR"))
install.packages("knitr")
library(knitr)

library(RColorBrewer)
library(gridBase)
library(ElemStatLearn)
install.packages("foreign")
library(foreign)
library(Hmisc)
library(tree)
library(rpart)
library(maptree)
## Loading required package: cluster
# install.packages('class') #not work in 2.15.2 library(class) #k Nearest
# Neighbors
library(e1071)  # Support Vector Machine
## Loading required package: class
library(ROCR)
library(ggplot2)
library(gtools)
library(caTools)
library(KernSmooth)
library( 'gplots')

library(grid)

##Exploratory analysis
#Look at the data set
head(trainf)
dim(trainf)
str(trainf)
summary(trainf)
colnames(trainf)
sapply(trainf[1, ], class)
#Find number of missing values/check ranges
sum(is.na(trainf))
#The Class Label
class(trainf$Spam)
##Numbers of correct Email in Dataset
result <- table(train$spam)
result
numEmail <- result[["non-spammer"]]
numEmail

print(paste0("Percentage: ", round((numEmail/nrow(train)) * 100, 2), "%"))
##Numbers of Spam in Dataset
numSpam <- result[["spammer"]]
numSpam
print(paste0("Percentage: ", round((numSpam/nrow(train)) * 100, 2), "%"))
##Numbers of Email vs. Numbers of Spam in DataSet
CUSTOM_COLORS_PLOT <- colorRampPalette(brewer.pal(10, "Set3"))
resTable <- table(train$spam)
par(mfrow = c(1, 2))
par(mar = c(5, 4, 4, 2) + 0.1)  # increase y-axis margin.
plot <- plot(train$spam, col = CUSTOM_COLORS_PLOT(2), main = "Email vs. Spam", 
             ylim = c(0, 4000), ylab = "Examples Number")
text(x = plot, y = resTable + 200, labels = resTable)
percentage <- round(resTable/sum(resTable) * 100)
labels <- paste(row.names(resTable), percentage)  # add percents to labels
labels <- paste(labels, "%", sep = "")  # ad % to labels
pie(resTable, labels = labels, col = CUSTOM_COLORS_PLOT(2), main = "Email vs. Spam")

#Average percentage factors
dataset.email <- sapply(test2[which(test2$spam == "non-spammer"), 1:14], function(x) ifelse(is.numeric(x), 
                                                                                          round(mean(x), 2), NA))
dataset.spam <- sapply(test2[which(test2$spam == "spammer"), 1:14], function(x) ifelse(is.numeric(x), 
                                                                                        round(mean(x), 2), NA))

dataset.email.order <- dataset.email[order(-dataset.email)[1:14]]
dataset.spam.order <- dataset.spam[order(-dataset.spam)[1:14]]

par(mfrow = c(1, 2))
par(mar = c(8, 4, 4, 2) + 0.1)  # increase y-axis margin.
plot <- barplot(dataset.email.order, col = CUSTOM_COLORS_PLOT(10), main = "Email: Average Percentage", 
                names.arg = "", ylab = "Percentage Relative (%)")
# text(x=plot,y=dataset.email.order-0.1, labels=dataset.email.order,
# cex=0.6)
vps <- baseViewports()
pushViewport(vps$inner, vps$figure, vps$plot)
grid.text(names(dataset.email.order), x = unit(plot, "native"), y = unit(-1, 
                                                                         "lines"), just = "right", rot = 50)
popViewport(3)

plot <- barplot(dataset.spam.order, col = CUSTOM_COLORS_PLOT(10), main = "Spam: Average Percentage", 
                names.arg = "", ylab = "Percentage Relative (%)")
# text(x=plot,y=dataset.spam.order-0.1, labels=dataset.spam.order,
# cex=0.6)
vps <- baseViewports()
pushViewport(vps$inner, vps$figure, vps$plot)
grid.text(names(dataset.spam.order), x = unit(plot, "native"), y = unit(-1, 
                                                                        "lines"), just = "right", rot = 50)
popViewport(3)

#TRAINING and TESTING data set for Classification
#Test data set
# Email vs. Spam
resTable <- table(test2$spam)
par(mfrow = c(1, 1))
par(mar = c(5, 4, 4, 2) + 0.1)  # increase y-axis margin.
plot <- plot(test2$spam, col = CUSTOM_COLORS_PLOT(6), main = "Email vs. Spam (Training Data Set)", 
             ylim = c(0, max(resTable) + 100), ylab = "Examples Number")
text(x = plot, y = resTable + 50, labels = resTable, cex = 0.75)
par(mfrow = c(1, 1))
percentage <- round(resTable/sum(resTable) * 100)
labels <- paste0(row.names(resTable), " (", percentage, "%) ")  # add percents to labels
pie(resTable, labels = labels, col = CUSTOM_COLORS_PLOT(10), main = "Email vs. Spam (Testing Data Set)")



#Training data set
# Email vs. Spam
resTable <- table(train$spam)
par(mfrow = c(1, 1))
par(mar = c(5, 4, 4, 2) + 0.1)  # increase y-axis margin.
plot <- plot(train$spam, col = CUSTOM_COLORS_PLOT(6), main = "Email vs. Spam (Traininging Data Set)", 
             ylim = c(0, max(resTable) + 100), ylab = "Examples Number")
text(x = plot, y = resTable + 50, labels = resTable, cex = 0.75)
par(mfrow = c(1, 1))
percentage <- round(resTable/sum(resTable) * 100)
labels <- paste0(row.names(resTable), " (", percentage, "%) ")  # add percents to labels
pie(resTable, labels = labels, col = CUSTOM_COLORS_PLOT(10), main = "Email vs. Spam (Trainging Data Set)")

#Classification. Predictive Model. RPart (Recursive Partitioning and Regression Trees) Algorithm

pc <- proc.time()
model.rpart <- rpart(spam ~ ., method = "class", data = train)
proc.time() - pc
printcp(model.rpart)

plot(model.rpart, uniform = TRUE, main = "Classification (RPART). Classification Tree for SPAM")
text(model.rpart, all = TRUE, cex = 0.75)
draw.tree(model.rpart, cex = 0.5, nodeinfo = TRUE, col = gray(0:8/8))
##Confusion Matrix (RPart)
prediction.rpart <- predict(model.rpart, newdata = test2, type = "class")
table(`Actual Class` = test2$spam, `Predicted Class` = prediction.rpart)
error.rate.rpart <- sum(test2$spam != prediction.rpart)/nrow(test2)
print(paste0("Accuary (Precision): ", 1 - error.rate.rpart))

#Classification. Predictive Model. SVM (Support Vector Machine) Algorithm
pc <- proc.time()
model.svm <- svm(spam ~ ., method = "class", data = train)
proc.time() - pc
summary(model.svm)
##Confusion Matrix (SVM)
prediction.svm <- predict(model.svm, newdata = test2, type = "class")
table(`Actual Class` = test2$spam, `Predicted Class` = prediction.svm)
error.rate.svm <- sum(test2$spam != prediction.svm)/nrow(test2)
print(paste0("Accuary (Precision): ", 1 - error.rate.svm))





####Model Evalution#########
## Predictor variable importance
install.packages("MachineShop")
library(MachineShop)
## Analysis libraries
library(MachineShop)
library(survival)
library(MASS)
library(magrittr)
## All available models
modelinfo() %>% names

#Preprocessing Recipe Naive Bias
## Recipe specification
library(recipes)

rec <- recipe(Spam ~ ., data = trainf)

Naive<-fit(rec, model = NaiveBayesModel)
Naive

## Spam performance metrics

## Observed responses
obs <- response(Naive, newdata = testf2)

## Predicted Spam means
pred_means <- predict(Naive, newdata = testf2)
performance(obs, pred_means)
#Accuracy       Kappa Sensitivity Specificity 
#0.9970000   0.9692938   1.0000000   0.9968421 


