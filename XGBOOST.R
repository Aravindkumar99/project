## Libraries:

library(data.table)
library(randomForest)
library(ggplot2)
library(caret)
library(dplyr)
library(xgboost)
library(ROCR)
library(pROC)

## READING FILES ##

train = fread("train.csv")
test = fread("test.csv")

## PERCENTAGE OF TARGET VALUES FACTORS ##
train$target = as.factor(train$target)

train %>% select(target) %>% group_by(target) %>% summarise(count = n()) %>% 
  ggplot(aes(x=target, y=count, fill= target)) +geom_bar(stat = 'identity')


## FINDING MISSING VALUES ##
sapply(train, function(x){round(sum(x==-1)*100/nrow(train),2)})
sapply(test,function(x){round(sum(x==-1)*100/nrow(train),2)})


## REPLACING MISSING VALUES ##

#ps_car_03_cat and ps_car_05_cat have a large proportion of records with missing values. Remove these variables.
#For the other categorical variables with missing values, we can leave the missing value -1 as such.
#ps_reg_03 (continuous) has missing values for 18% of all records. Replace by the mean.
#ps_car_11 (ordinal) has only 5 records with misisng values. Replace by the mode.
#ps_car_12 (continuous) has only 1 records with missing value. Replace by the mean.
#ps_car_14 (continuous) has missing values for 7% of all records. Replace by the mean.


## ON TRAIN DATA:
train = train[,-c("ps_car_03_cat","ps_car_05_cat")]
train[which(train$ps_reg_03 == -1), 'ps_reg_03']= mean(train$ps_reg_03)
train[which(train$ps_car_11==-1), 'ps_car_11'] = as.numeric(names(sort(-table(train$ps_car_11))))[1]
train[which(train$ps_car_14 == -1),'ps_car_14'] = mean(train$ps_car_14)
train[which(train$ps_car_12 == -1),'ps_car_12'] = mean(train$ps_car_12)


### ON TEST DATA:
test = test[,-c("ps_car_03_cat","ps_car_05_cat")]
test[which(test$ps_reg_03 == -1), 'ps_reg_03']= mean(test$ps_reg_03)
test[which(test$ps_car_11==-1), 'ps_car_11'] = as.numeric(names(sort(-table(test$ps_car_11))))[1]
test[which(test$ps_car_14 == -1),'ps_car_14'] = mean(test$ps_car_14)
test[which(test$ps_car_12 == -1),'ps_car_12'] = mean(test$ps_car_12)


## Under sampling: here we reduce the number of zeros to number equal to ones.

train_ins_0<-train%>%filter(target==0)
dim(train_ins_0)

train_ins_1<-train%>%filter(target==1)
dim(train_ins_1)

df_class0_under<-train_ins_0[sample(nrow(train_ins_1)),]
dim(df_class0_under)

final_train_ins<-rbind(train_ins_1,df_class0_under)
table(final_train_ins$target)     ### forms a dataframe with equal no. of ones and zeros


## XG_boosting##

#-eta: It is also known as the learning rate or the shrinkage factor. It actually shrinks the feature weights to make the boosting process more conservative. The range is 0 to 1. Low eta value means the model is more robust to overfitting.
#-gamma: The range is 0 to ???. Larger the gamma more conservative the algorithm is.
#-max_depth: We can specify maximum depth of a tree using this parameter.
#-subsample: It is the proportion of rows that the model will randomly select to grow trees.
#-colsample_bytree: It is the ratio of variables randomly chosen to build each tree in the model.


param_list=list(
  objective= "binary:logistic",
  eta=0.01,
  gamma=1,
  max_depth=6,
  subsample=0.8,
  colsample_bytree=0.5
)


instrain = xgb.DMatrix(data=as.matrix(final_train_ins[,-c(1,2)]),label=as.numeric(as.character(final_train_ins$target)))

## matrix conversion
## Crossvalidation
#-We are going to use the xgb.cv() function for cross validation. Here we are using cross validation for finding the optimal value of nrounds.
#-As per the verbose above, we got the best validation/test score at the 430th iteration. Hence, we will use nrounds = 165 for building the XGBoost model.

set.seed(108)
xgbcv=xgb.cv(params = param_list,
             data=instrain,
             nrounds = 1000,
             nfold = 5,
             print_every_n = 10,
             early_stopping_rounds = 30,
             maximize = F)


# Model training
#- As per the verbose above, we got the best validation/test score at the 184th iteration. Hence, we will use nrounds = 184 for building the XGBoost model.

xgb_model=xgb.train(data=instrain,params = param_list,nrounds = 116)
xgb_model

# Model Testing

instest = xgb.DMatrix(data=as.matrix(test[,-1]))
target = predict(xgb_model,instest)
result = as.data.frame.matrix(cbind(test$id,target))
nrow(result)
names(result)= c("id", "target")
result$id = as.integer(result$id)
write.csv(result,"result.csv", row.names = FALSE)

###################################### MODEL RESULTS FOR AUC, KAPPA AND F1 SCORE FOR BALANCED DATA #############################

## Splitting Train dataset into Train and Test

train_sample = final_train_ins[sample(c(1:nrow(final_train_ins)),0.9*nrow(final_train_ins)),]
test_sample = final_train_ins[-sample(c(1:nrow(final_train_ins)),0.9*nrow(final_train_ins)),]

class(train_sample$target)

## Model building on Train_Sample
train_sample_mat = xgb.DMatrix(data = as.matrix(train_sample[,-c(1,2)]), label=as.numeric(as.character(train_sample$target)))
dim(train_sample_mat)

set.seed(123)
xgbcv_train=xgb.cv(params = param_list,
                   data=train_sample_mat,
                   nrounds = 1000,
                   nfold = 5,
                   print_every_n = 10,
                   early_stopping_rounds = 30,
                   maximize = F)
xgb_train=xgb.train(data=train_sample_mat,params = param_list,nrounds = 168)
xgb_train

# Testing on Test Data:
test_sample_train = xgb.DMatrix(data=as.matrix(test_sample[,-c(1,2)]))
dim(test_sample_train)
target = predict(xgb_train,test_sample_train)
target_values = as.factor(ifelse(target>0.5,1,0))
### AUC FOR THE MODEL:

xg_roc = roc(test_sample$target,target)
a1 = auc(xg_roc)
plot(xg_roc)

cf = confusionMatrix(target_values,test_sample$target,positive = '1')
cf$overall['Accuracy']
cf$byClass['F1']
cf$overall['Kappa']

