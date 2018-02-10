# Practical Machine Learning Course Project
Intan Dea Yutami  
10 Februari 2018  



## Executive Summary

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

### Data

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


The data for this project come from this source: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

## Preliminary Step

### Load The Required Package/Library

Below is the code to load the required library


```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(rpart)
library(rattle)
```

```
## Rattle: A free graphical interface for data science with R.
## Versi 5.1.0 Copyright (c) 2006-2017 Togaware Pty Ltd.
## Ketik 'rattle()' untuk shake, rattle, dan roll data.
```

```r
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:rattle':
## 
##     importance
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
library(parallel)
library(doParallel)
```

```
## Loading required package: foreach
```

```
## Loading required package: iterators
```

### Load and Clean Up The Training and Testing Datasets

In this chunk of code below, there are two CSV files read. pml-training and pml-testing. pml-training will be separated later into training and testing data sets, while pml-testing will be used later for quiz.

The columns that having NA values, empty value, or DIV/0 are removed. The first seven columns are also removed because it is not needed for model fitting usage (because they are only names, timestamps, etc.)

```r
trainingSet <- read.csv("pml-training.csv", header=TRUE)
vldSet <- read.csv("pml-testing.csv", header=TRUE)
trainingSet <- trainingSet[sapply(trainingSet, function(x) !any(is.na(x) || x == "" || x == "#DIV/0!"))][,-1:-7]
quizSet <- vldSet[sapply(vldSet, function(x) !any(is.na(x)|| x == "" || x == "#DIV/0!"))][,-1:-7]
```

### Partition the Training Set 

The training variable now is partitioned into training and test sets. Getting rid of columns that its variance is near zero is also done below.

```r
inTrain <- createDataPartition(y = trainingSet$classe, p = 0.75, list = FALSE)
training <- trainingSet[inTrain,]
testing <- trainingSet[-inTrain,]

nzv <- nearZeroVar(training, saveMetrics=TRUE)
training <- training[,nzv$nzv==FALSE]

nzv<- nearZeroVar(testing,saveMetrics=TRUE)
testing <- testing[,nzv$nzv==FALSE]

dim(training)
```

```
## [1] 14718    53
```

```r
dim(testing)
```

```
## [1] 4904   53
```
## Fitting Model and Predicting

In this project random forest method is chosen as method for training. 

```r
set.seed(100)

#start doing parallel computing
cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster)

#fitting model
fitControl <- trainControl(method = "cv", number = 5, allowParallel = TRUE)
rfMod <- train(classe ~ ., data = training, trControl = fitControl, method = "rf")
plot(rfMod)
```

![](PML_files/figure-html/unnamed-chunk-4-1.png)<!-- -->

```r
#prediction
pred_rf <- predict(rfMod, testing)
cm_rf <- confusionMatrix(pred_rf, testing$classe)
cm_rf
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    2    0    0    0
##          B    0  946    7    0    0
##          C    0    1  846    8    0
##          D    0    0    2  796    2
##          E    0    0    0    0  899
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9955          
##                  95% CI : (0.9932, 0.9972)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9943          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9968   0.9895   0.9900   0.9978
## Specificity            0.9994   0.9982   0.9978   0.9990   1.0000
## Pos Pred Value         0.9986   0.9927   0.9895   0.9950   1.0000
## Neg Pred Value         1.0000   0.9992   0.9978   0.9981   0.9995
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1929   0.1725   0.1623   0.1833
## Detection Prevalence   0.2849   0.1943   0.1743   0.1631   0.1833
## Balanced Accuracy      0.9997   0.9975   0.9936   0.9945   0.9989
```

According to the prediction result, its accuracy is 99.53%. The expected out of sample error is 1 - 99.57% = 0.03%. Because of this good result, random forest is still used to predict 20 samples for the quiz submission.


```r
#stop doing parallel computing
stopCluster(cluster)
registerDoSEQ()
```

# Testing for Quiz Project

Below is the code and the result of testing 20 samples for quiz submission

```r
pred_quiz <-predict(rfMod, quizSet)
pred_quiz
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
