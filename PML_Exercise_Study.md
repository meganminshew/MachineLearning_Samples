###Practical Machine Learning Research Project
* author: Megan Minshew
* date: August 21, 2014
* output: PML_Exercise_Study.Rmd
* html_document:PML_Exercise_Study.html


###Study Summary
This study attempts to classify the manner in which a person exercises, based on data collected by quantified self movement devices, such as Jawbone Up, Nike FuelBand and Fitbit. The study participants were instructed to perform a series of exercises five different ways with only one way being the correct way. Data provided by the study classifies the exercise in the following categories: A-exactly according to specification, B-throwing elbows to the front, C-lifting the dumbbell halfway, D-lowering the dumbbell halfway, and E-throwing hips to the front.

The purpose of this study is to develop the means to improve exercise by incorporating a feedback mechanism into the tracking device, warning a person if they are exercising in a less than optimal manner. Assessing the correctness of the categorization of exercise is required before the feasability of the product can be measured. 

Data for this study is provided by and available at:
http://groupware.les.inf.puc-rio.br/har



```r
#using libraries:
library(kernlab); library(psych); library(rattle); library(rpart); library(caret); library(randomForest)
```

###Exploratory Analysis
Exercise measurement data is loaded and split into 60% training and 40% model testing. Analysis of the 160 data elements available indicate a number of variables to be excluded because they are sparsely populated, contain no variation in the data or were not measures of movement. This study assumes that subject and time specific indicators should be ignored.


```r
#load the activity data
data <- read.csv("pml-training.csv") #training data provided by the study

inTrain <- createDataPartition(y=data$classe,p=0.60, list=FALSE)
training <- data[inTrain,]
testing <- data[-inTrain,]

#Survey the data to identify useful elements
d <- describe(training)
d[8:20,2:8]
```

```
##                           n   mean    sd median trimmed    mad    min
## roll_belt             11776  64.45 62.75 114.00   63.47  60.79  -28.6
## pitch_belt            11776   0.26 22.40   5.31    2.38  10.52  -54.9
## yaw_belt              11776 -11.05 95.38 -12.60  -22.98 111.79 -180.0
## total_accel_belt      11776  11.32  7.74  17.00   11.13  11.86    0.0
## kurtosis_roll_belt*   11776   5.04 32.02   1.00    1.00   0.00    1.0
## kurtosis_picth_belt*  11776   4.46 27.04   1.00    1.00   0.00    1.0
## kurtosis_yaw_belt*    11776   1.02  0.14   1.00    1.00   0.00    1.0
## skewness_roll_belt*   11776   5.10 32.13   1.00    1.00   0.00    1.0
## skewness_roll_belt.1* 11776   4.54 27.63   1.00    1.00   0.00    1.0
## skewness_yaw_belt*    11776   1.02  0.14   1.00    1.00   0.00    1.0
## max_roll_belt           248  -5.73 93.58  -4.90  -16.45 122.76  -94.1
## max_picth_belt          248  13.16  8.03  18.00   12.88  11.86    3.0
## max_yaw_belt*         11776   1.33  2.92   1.00    1.00   0.00    1.0
```

```r
#Remove the attributes that aren't fully populated
d <- d[d$n==max(d$n),]
#d$mad has a number of attributes with zero variance, remove those
d <- d[d$mad!=0,]
#remove the counters, participant and time attributes
d <- d[7:59,]
#subset the training data to the columns with significant data
t <- training[,d$vars]
```
###Machine Learning Evaluations
The classification of the quality of exercise is based on 52 measures of movement in one or more components of 3-D space. Linear modeling and generalized linear modeling have been eliminated as the data will not comform to the algorithm design.

Decision Trees, Bagging and Random Forests are appropriate algorithms for handling the study data and will be evaluated and compared.

Measurement can be roughly sorted into one of four categories: belt, arm, dumbell and forearm. Each category contains 13 different measures. To limit the processing time, cross-validation passes will be a third of the number of measures per category or 5 for algorithms that use cross-validation.


```r
#Set the cvLimit value
cvLimit = trainControl(method="cv", number = 5)
```

###Decision Trees or Recursive Partitioning method

```r
#get a test and train set for decision trees
r_t <- t
r_testing <- testing
r_model <- train(classe ~ ., method="rpart", data = r_t)
#Show the trees
fancyRpartPlot(r_model$finalModel)
```

![plot of chunk unnamed-chunk-4](figure/unnamed-chunk-4.png) 

```r
#Summarize the model
r_model$finalModel
```

```
## n= 11776 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 11776 8428 A (0.28 0.19 0.17 0.16 0.18)  
##    2) roll_belt< 130.5 10807 7468 A (0.31 0.21 0.19 0.18 0.11)  
##      4) pitch_forearm< -33.65 958    9 A (0.99 0.0094 0 0 0) *
##      5) pitch_forearm>=-33.65 9849 7459 A (0.24 0.23 0.21 0.2 0.12)  
##       10) magnet_dumbbell_y< 433.5 8303 5964 A (0.28 0.18 0.24 0.19 0.11)  
##         20) roll_forearm< 123.5 5163 3067 A (0.41 0.18 0.19 0.17 0.061) *
##         21) roll_forearm>=123.5 3140 2112 C (0.077 0.18 0.33 0.23 0.19) *
##       11) magnet_dumbbell_y>=433.5 1546  757 B (0.033 0.51 0.041 0.22 0.19) *
##    3) roll_belt>=130.5 969    9 E (0.0093 0 0 0 0.99) *
```

```r
#Predict the testing data from the training data model
r_predicted <- predict(r_model, newdata=r_testing[,-1])
#Display the results
table(r_predicted, r_testing$classe)
```

```
##            
## r_predicted    A    B    C    D    E
##           A 2017  629  620  567  202
##           B   55  527   51  250  204
##           C  155  362  697  469  365
##           D    0    0    0    0    0
##           E    5    0    0    0  671
```

```r
#How accurate is the prediction
confusionMatrix(r_predicted,r_testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2017  629  620  567  202
##          B   55  527   51  250  204
##          C  155  362  697  469  365
##          D    0    0    0    0    0
##          E    5    0    0    0  671
## 
## Overall Statistics
##                                        
##                Accuracy : 0.499        
##                  95% CI : (0.487, 0.51)
##     No Information Rate : 0.284        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.345        
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.904   0.3472   0.5095    0.000   0.4653
## Specificity             0.641   0.9115   0.7914    1.000   0.9992
## Pos Pred Value          0.500   0.4848   0.3403      NaN   0.9926
## Neg Pred Value          0.944   0.8534   0.8843    0.836   0.8925
## Prevalence              0.284   0.1935   0.1744    0.164   0.1838
## Detection Rate          0.257   0.0672   0.0888    0.000   0.0855
## Detection Prevalence    0.514   0.1385   0.2610    0.000   0.0862
## Balanced Accuracy       0.772   0.6293   0.6505    0.500   0.7323
```

###Boosting

```r
#get a test and train set for bagging
b_t <- t
b_testing <- testing
b_model <- train(classe ~ ., method="gbm", trControl=cvLimit, data=b_t, verbose=FALSE)
#Summarize the model
summary(b_model)
```

![plot of chunk unnamed-chunk-5](figure/unnamed-chunk-5.png) 

```
##                                       var  rel.inf
## roll_belt                       roll_belt 22.06972
## pitch_forearm               pitch_forearm 10.61799
## yaw_belt                         yaw_belt  8.48938
## magnet_dumbbell_z       magnet_dumbbell_z  7.13584
## magnet_dumbbell_y       magnet_dumbbell_y  5.05192
## magnet_belt_z               magnet_belt_z  4.98197
## roll_forearm                 roll_forearm  4.82124
## pitch_belt                     pitch_belt  4.33140
## roll_dumbbell               roll_dumbbell  3.13392
## accel_forearm_x           accel_forearm_x  2.61313
## gyros_belt_z                 gyros_belt_z  2.59544
## gyros_dumbbell_y         gyros_dumbbell_y  2.10195
## accel_dumbbell_y         accel_dumbbell_y  1.99770
## magnet_forearm_z         magnet_forearm_z  1.94854
## accel_forearm_z           accel_forearm_z  1.68205
## yaw_arm                           yaw_arm  1.63274
## accel_dumbbell_x         accel_dumbbell_x  1.62039
## roll_arm                         roll_arm  1.29462
## magnet_arm_x                 magnet_arm_x  1.13108
## magnet_belt_x               magnet_belt_x  1.06453
## magnet_arm_z                 magnet_arm_z  1.04316
## magnet_dumbbell_x       magnet_dumbbell_x  0.87616
## magnet_belt_y               magnet_belt_y  0.78494
## magnet_forearm_x         magnet_forearm_x  0.74875
## accel_dumbbell_z         accel_dumbbell_z  0.67596
## total_accel_dumbbell total_accel_dumbbell  0.59711
## magnet_arm_y                 magnet_arm_y  0.51714
## gyros_arm_y                   gyros_arm_y  0.47169
## gyros_belt_y                 gyros_belt_y  0.46052
## pitch_dumbbell             pitch_dumbbell  0.41467
## accel_belt_z                 accel_belt_z  0.34367
## yaw_dumbbell                 yaw_dumbbell  0.32871
## total_accel_arm           total_accel_arm  0.29931
## total_accel_forearm   total_accel_forearm  0.28696
## accel_forearm_y           accel_forearm_y  0.28522
## accel_arm_x                   accel_arm_x  0.26678
## gyros_forearm_z           gyros_forearm_z  0.23693
## accel_arm_y                   accel_arm_y  0.22947
## gyros_dumbbell_x         gyros_dumbbell_x  0.18896
## accel_arm_z                   accel_arm_z  0.17483
## magnet_forearm_y         magnet_forearm_y  0.13908
## yaw_forearm                   yaw_forearm  0.10752
## total_accel_belt         total_accel_belt  0.08681
## gyros_dumbbell_z         gyros_dumbbell_z  0.06946
## gyros_arm_x                   gyros_arm_x  0.05066
## gyros_belt_x                 gyros_belt_x  0.00000
## accel_belt_x                 accel_belt_x  0.00000
## accel_belt_y                 accel_belt_y  0.00000
## pitch_arm                       pitch_arm  0.00000
## gyros_arm_z                   gyros_arm_z  0.00000
## gyros_forearm_x           gyros_forearm_x  0.00000
## gyros_forearm_y           gyros_forearm_y  0.00000
```

```r
#Predict the testing data from the training data model
b_predicted <- predict(b_model, newdata=b_testing[,-1])
#Display the results
table(b_predicted, b_testing$classe)
```

```
##            
## b_predicted    A    B    C    D    E
##           A 2195   50    0    2    1
##           B   26 1422   49    6   20
##           C    8   43 1293   35   13
##           D    2    2   24 1231   14
##           E    1    1    2   12 1394
```

```r
#How accurate is the prediction
confusionMatrix(b_predicted, b_testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2195   50    0    2    1
##          B   26 1422   49    6   20
##          C    8   43 1293   35   13
##          D    2    2   24 1231   14
##          E    1    1    2   12 1394
## 
## Overall Statistics
##                                         
##                Accuracy : 0.96          
##                  95% CI : (0.956, 0.965)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.95          
##  Mcnemar's Test P-Value : 1.82e-06      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.983    0.937    0.945    0.957    0.967
## Specificity             0.991    0.984    0.985    0.994    0.998
## Pos Pred Value          0.976    0.934    0.929    0.967    0.989
## Neg Pred Value          0.993    0.985    0.988    0.992    0.993
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.280    0.181    0.165    0.157    0.178
## Detection Prevalence    0.287    0.194    0.177    0.162    0.180
## Balanced Accuracy       0.987    0.960    0.965    0.975    0.982
```

###Random Forests

```r
#get a test and train set for Decision Trees
rf_t <- t
rf_testing <- testing
rf_model <- train(classe ~ ., method="rf", trControl=cvLimit, data=rf_t, prox=TRUE)
#Summarize the model
summary(rf_model)
```

```
##                 Length    Class      Mode     
## call                    5 -none-     call     
## type                    1 -none-     character
## predicted           11776 factor     numeric  
## err.rate             3000 -none-     numeric  
## confusion              30 -none-     numeric  
## votes               58880 matrix     numeric  
## oob.times           11776 -none-     numeric  
## classes                 5 -none-     character
## importance             52 -none-     numeric  
## importanceSD            0 -none-     NULL     
## localImportance         0 -none-     NULL     
## proximity       138674176 -none-     numeric  
## ntree                   1 -none-     numeric  
## mtry                    1 -none-     numeric  
## forest                 14 -none-     list     
## y                   11776 factor     numeric  
## test                    0 -none-     NULL     
## inbag                   0 -none-     NULL     
## xNames                 52 -none-     character
## problemType             1 -none-     character
## tuneValue               1 data.frame list     
## obsLevels               5 -none-     character
```

```r
#Predict the testing data from the training data model
rf_predicted <- predict(rf_model, newdata=rf_testing[,-1])
#Display the results
table(rf_predicted, rf_testing$classe)
```

```
##             
## rf_predicted    A    B    C    D    E
##            A 2229   13    0    0    0
##            B    3 1501    9    1    0
##            C    0    4 1355   13    5
##            D    0    0    4 1271    3
##            E    0    0    0    1 1434
```

```r
#How accurate is the prediction
confusionMatrix(rf_predicted, rf_testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2229   13    0    0    0
##          B    3 1501    9    1    0
##          C    0    4 1355   13    5
##          D    0    0    4 1271    3
##          E    0    0    0    1 1434
## 
## Overall Statistics
##                                         
##                Accuracy : 0.993         
##                  95% CI : (0.991, 0.995)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.991         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.999    0.989    0.990    0.988    0.994
## Specificity             0.998    0.998    0.997    0.999    1.000
## Pos Pred Value          0.994    0.991    0.984    0.995    0.999
## Neg Pred Value          0.999    0.997    0.998    0.998    0.999
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.191    0.173    0.162    0.183
## Detection Prevalence    0.286    0.193    0.176    0.163    0.183
## Balanced Accuracy       0.998    0.993    0.994    0.994    0.997
```

###Study Findings
The 99.9% Balanced accuracy at indentifing Class A or correctly performed exercise obtained by the Random Forest model is slightly better than Boostings 98.9%. Comparing the confusion matrix from all three models supports this choice across all other metrics, particularly with the overall accuracy and the 95% confidence interval for the models predictions. The following Accuracy for cross-validation by model chart supports the choice of the Random Forest model. 


```r
par(mfrow=c(3,1))
plot(rf_model, metric='Accuracy', main="Random Forest Model Accuracy")
```

![plot of chunk unnamed-chunk-7](figure/unnamed-chunk-71.png) 

```r
plot(b_model, metric='Accuracy', main="Boosting Model Accuracy")
```

![plot of chunk unnamed-chunk-7](figure/unnamed-chunk-72.png) 

```r
plot(r_model, metric='Accuracy', main="Decision Trees Model Accuracy")
```

![plot of chunk unnamed-chunk-7](figure/unnamed-chunk-73.png) 


As a final step this study will predict the exercise accuracy class from a sample of 20 rows of data that is independent of the data that produced the model.


```r
testData <- read.csv("pml-testing.csv") #testing data for final model proof
finalPredict <- predict(rf_model, testData)
finalPredict
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
