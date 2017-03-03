# Practical Machine Learning Project
Ramesh Ramiah  
March 3, 2017  
##1. Background  
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.  
  
The goal of this project, is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict if they performed the activity correctly or incorrectly in 5 different ways (variable "classe"). Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes.  
  
##2. Data Processing  
###2.1 Data and Documentation Source  
The training data used for this analysis is available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)  
  
The test data used for this analysis is available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)  
  
The documentation for this analysis is available [here](http://groupware.les.inf.puc-rio.br/har#ixzz4aEDRLc7c)  
  
###2.2 Download and read the data set  
2.2.1 Set global options and load required libraries for the analysis.    

```r
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(corrplot)
library(reshape2)
library(knitr)
opts_chunk$set(echo = TRUE, results = "hold", fig.width=12, fig.height=8, warning=FALSE, message=FALSE)  
```
  
2.2.2 The data was downloaded and read into a csv file.  

```r
TrainUrl <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
TestUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
TrainFile <- "./Data/pml-training.csv"
TestFile <- "./Data/pml-testing.csv"
    
if (!file.exists("./Data")) {
    dir.create("./Data")
}
if (!file.exists(TrainFile)) {
    download.file(TrainUrl, destfile=TrainFile)
}
if (!file.exists(TestFile)) {
    download.file(TestUrl, destfile=TestFile)
}
training <- read.csv("./Data/pml-training.csv", header = T, 
                        na.strings = c("NA","#DIV/0!",""))  
testing <- read.csv("./Data/pml-testing.csv", header = T, 
                         na.strings = c("NA","#DIV/0!",""))
```
  
##3. Exploratory Data Analysis  
###3.1 Check the dimensions of data frame.  

```r
dim(training);dim(testing)
```

```
## [1] 19622   160
## [1]  20 160
```
  
The training data set consist of 19622 observations and 160 variables, whereas the testing data set has the same number of variables and 20 observations.  
  
###3.2 Clean the data  
The data set were check for any missing values.  

```r
sum(is.na(training))
sum(is.na(testing))
```

```
## [1] 1925102
## [1] 2000
```
  
Remove columns (predictors) of the training set that contain any missing values.  

```r
training <- training[, colSums(is.na(training)) == 0]
testing <- testing[, colSums(is.na(testing)) == 0]
```
  
View the names of the variables  

```r
names(training)
```

```
##  [1] "X"                    "user_name"            "raw_timestamp_part_1"
##  [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
##  [7] "num_window"           "roll_belt"            "pitch_belt"          
## [10] "yaw_belt"             "total_accel_belt"     "gyros_belt_x"        
## [13] "gyros_belt_y"         "gyros_belt_z"         "accel_belt_x"        
## [16] "accel_belt_y"         "accel_belt_z"         "magnet_belt_x"       
## [19] "magnet_belt_y"        "magnet_belt_z"        "roll_arm"            
## [22] "pitch_arm"            "yaw_arm"              "total_accel_arm"     
## [25] "gyros_arm_x"          "gyros_arm_y"          "gyros_arm_z"         
## [28] "accel_arm_x"          "accel_arm_y"          "accel_arm_z"         
## [31] "magnet_arm_x"         "magnet_arm_y"         "magnet_arm_z"        
## [34] "roll_dumbbell"        "pitch_dumbbell"       "yaw_dumbbell"        
## [37] "total_accel_dumbbell" "gyros_dumbbell_x"     "gyros_dumbbell_y"    
## [40] "gyros_dumbbell_z"     "accel_dumbbell_x"     "accel_dumbbell_y"    
## [43] "accel_dumbbell_z"     "magnet_dumbbell_x"    "magnet_dumbbell_y"   
## [46] "magnet_dumbbell_z"    "roll_forearm"         "pitch_forearm"       
## [49] "yaw_forearm"          "total_accel_forearm"  "gyros_forearm_x"     
## [52] "gyros_forearm_y"      "gyros_forearm_z"      "accel_forearm_x"     
## [55] "accel_forearm_y"      "accel_forearm_z"      "magnet_forearm_x"    
## [58] "magnet_forearm_y"     "magnet_forearm_z"     "classe"
```
  
Since the analysis requires to use data from accelerometers on the belt, forearm, arm, and dumbell, columns 1 to 7 will be removed from both data sets.  

```r
training <- training[, -c(1:7)]
testing <- testing[, -c(1:7)]
```
  
###3.3 Relationship of variables and classe  
A plot was created to show the relationship between the variables and classes. From the plot below, each features relatively has the same distribution among the 5 outcome levels (A, B, C, D, E).  

```r
featurePlot(x=training[,1:52], y=training$classe, plot = "strip")
```
  
![](./Plots/PML_Plot1.png)  
  
###3.4 Multicollinearity  
A correlation matrix of the variable was done to determine if the variable are highly correlated to each other.  

```r
CorPlot <- cor(training[,-length(names(training))])
MeltCorPlot <- melt(CorPlot)
## qplot(x=Var1, y=Var2, data=MeltCorPlot, fill=value, geom="tile") +
##     scale_fill_gradient2(limits=c(-1, 1)) +
##     theme(axis.text.x = element_text(angle=-90, vjust=0.5, hjust=0))
```
  
![](./Plots/PML_Plot3.png)    
  
Based on the plot above, the variables do not seem to be highly correlated. Therefore no variables will be removed.  
  
  
##4.0 Prediction Algorithms  
Two different prediction models will used and compared. The selection will be based on the high accuracy and low out-of-sample error.  
  
###4.1 Cross Validation  
The training data is randomly split into a smaller training set and a validation set. This is the simplest and most widely used method for estimating prediction
error.  

```r
set.seed(100)
inTrain1 <- createDataPartition(training$classe, p = 0.7, list = FALSE)
Strain <- training[inTrain1,]
Svalid <- training[-inTrain1,]
```
  

  
###4.2 Random Forest  
A k-fold cross validation will be used on the Strain set to select the optimal tuning parameters for the random forest model. A 3 fold cross validation will be used since the data set has 13737 observations and this will save some processing time.  

```r
kf <- trainControl(method = "cv", number = 3)
ModfitRF <- train(classe ~., data = Strain, method="rf", trControl = kf)
PredRF <- predict(ModfitRF, Svalid)
confusionMatrix(Svalid$classe, PredRF)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    2    0    0    0
##          B    8 1128    3    0    0
##          C    0    3 1019    4    0
##          D    0    0   16  948    0
##          E    0    0    3    2 1077
## 
## Overall Statistics
##                                          
##                Accuracy : 0.993          
##                  95% CI : (0.9906, 0.995)
##     No Information Rate : 0.2855         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9912         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9952   0.9956   0.9789   0.9937   1.0000
## Specificity            0.9995   0.9977   0.9986   0.9968   0.9990
## Pos Pred Value         0.9988   0.9903   0.9932   0.9834   0.9954
## Neg Pred Value         0.9981   0.9989   0.9955   0.9988   1.0000
## Prevalence             0.2855   0.1925   0.1769   0.1621   0.1830
## Detection Rate         0.2841   0.1917   0.1732   0.1611   0.1830
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9974   0.9966   0.9887   0.9952   0.9995
```
  
Based on the results above, the model accuracy is 99.3% and the out-of-sample error rate is 0.7%.  
  
###4.3 Decision Tree  
Similarly as above a 3 fold cross validation will be used.  

```r
ModfitRP <- train(classe ~., data = Strain, method="rpart", trControl = kf)
PredRP <- predict(ModfitRP, Svalid)
confusionMatrix(Svalid$classe, PredRP)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1493   33  118    0   30
##          B  506  369  264    0    0
##          C  504   35  487    0    0
##          D  438  170  356    0    0
##          E  152  141  287    0  502
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4845          
##                  95% CI : (0.4716, 0.4973)
##     No Information Rate : 0.5256          
##     P-Value [Acc > NIR] : 1               
##                                           
##                   Kappa : 0.3256          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.4827   0.4933  0.32209       NA   0.9436
## Specificity            0.9352   0.8501  0.87674   0.8362   0.8916
## Pos Pred Value         0.8919   0.3240  0.47466       NA   0.4640
## Neg Pred Value         0.6200   0.9201  0.78905       NA   0.9938
## Prevalence             0.5256   0.1271  0.25692   0.0000   0.0904
## Detection Rate         0.2537   0.0627  0.08275   0.0000   0.0853
## Detection Prevalence   0.2845   0.1935  0.17434   0.1638   0.1839
## Balanced Accuracy      0.7089   0.6717  0.59942       NA   0.9176
```
  
Based on the results above, the model accuracy is 48.45% and the out-of-sample error rate is 51.55%. The decision tree model does not predict the outcome "classe" accurately.  
  
Clearly the random forest model is the more accurate model to perform the predictions.  
    
###4.4 Re-train the selected model on full training data set.  
It is always good to re-train the selected model random forest on the full data set before predicting the test set.  

```r
set.seed(100)
ModfitRF1 <- train(classe ~., data = training, method="rf", trControl = kf)
```
  
##5.0 Predictions  
The Random Forest model (ModFitRF1) will be used to make the testing data set predicts.  

```r
PredRF1 <- predict(ModfitRF1, testing)
PredRF1
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
  
##6.0 Reference  
Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements. Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6.
  
