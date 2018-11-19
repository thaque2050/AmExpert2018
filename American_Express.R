library(lattice)
library(ParamHelpers)
library(grid)
library(DMwR)
library(xgboost)
library(mlr)
library(dplyr)
library(tidyverse)
library(MlBayesOpt)
library(Matrix)
library(rBayesianOptimization)
library(lubridate)


#The code is split into 5 sections: 
#1. Data Transformation 
#2: Hyperparamete Optimization (no need to run this as I have hard coded the results in the model parameter) 
#3: Build Model 
#4: Make Preduction 
#5: Write in CSV file for submission


train_data<-read.csv("train_amex/train.csv")
historical_data<-read.csv("train_amex/historical_user_logs.csv")
test_data<-read.csv("test_LNMuIYp/test.csv")
submission_data<-read.csv("sample_submission_2s8l9nF.csv")

#DATA TRANSFORMATION
#Data Transformation - Train
dtrain<-train_data
dtrain$DateTime<-as.POSIXct(as.character(dtrain$DateTime),"%Y-%m-%d %H:%M")
dtrain$weekday<-as.numeric(as.factor(weekdays(dtrain$DateTime)))
dtrain$hour<-hour(dtrain$DateTime)

dummy_product<-model.matrix(~product+0,data=dtrain)
dummy_gender<-model.matrix(~gender+0,data=dtrain)

dtrain<-data.frame(dtrain,dummy_product,dummy_gender)

dtrain$session_id<-NULL
dtrain$DateTime<-NULL
dtrain$user_id<-NULL
dtrain$product<-NULL
dtrain$gender<-NULL

dtrain<-dtrain[,c(1,2,3,4,5,6,7,8,9,11,12, 13,14,15,16,17,18,19,20,21,22,23,24,25,10)]


#Data Transformation - Test
dtest<-test_data
dtest$DateTime<-as.POSIXct(as.character(dtest$DateTime),"%Y-%m-%d %H:%M")
dtest$weekday<-as.numeric(as.factor(weekdays(dtest$DateTime)))
dtest$hour<-hour(dtest$DateTime)


dummy_product<-model.matrix(~product+0,data=dtest)
dummy_gender<-model.matrix(~gender+0,data=dtest)

dtest<-data.frame(dtest,dummy_product,dummy_gender)

dtest$session_id<-NULL
dtest$DateTime<-NULL
dtest$user_id<-NULL
dtest$product<-NULL
dtest$gender<-NULL


#HYPERPARAMETER OPTIMIZATION

#X_train<-as.matrix(dtrain[,1:24])
#Y_train<-dtrain[,25]
#train_matrix<-xgb.DMatrix(data=X_train, label=Y_train)


#cv_folds <- KFold(dtrain$is_click, nfolds = 5,stratified = TRUE, seed = 0)

#xgb_cv_bayes <- function(eta,gamma,colsample_bytree,max_delta_step,lambda,alpha,
#                         max_depth, min_child_weight, subsample) {
#  cv <- xgb.cv(params = list(booster = "gbtree",
#                             eta = eta,
#                             max_depth = max_depth,
#                             min_child_weight = min_child_weight,
#                             subsample = subsample, 
#                             colsample_bytree = colsample_bytree,
#                             lambda = lambda,
#                             alpha = alpha,
#                             gamma=gamma,
#                             max_delta_step=max_delta_step,
#                             objective = "binary:logistic",
#                             eval_metric = "auc"),
#               data = train_matrix, nrounds=105,folds = cv_folds, prediction = TRUE, 
#               showsd = TRUE,early_stopping_rounds = 5, maximize = TRUE, verbose = 0)
#  list(Score = cv$evaluation_log$test_auc_mean[cv$best_iteration],
#       Pred = cv$pred)
#}

#OPT_Res <- BayesianOptimization(xgb_cv_bayes,
#                                bounds = list(max_depth = c(0L,50L),
#                                              min_child_weight = c(0,50),
#                                              subsample = c(0, 1.0),
#                                              eta=c(0,1.0),
#                                              colsample_bytree = c(0,1.0),
#                                              lambda = c(0,1.0),
#                                              alpha = c(0,1.0),
#                                              gamma=c(0,50),
#                                              max_delta_step=c(0,50)),
#                                init_grid_dt = NULL, init_points = 10, n_iter = 60,
#                                acq = "ucb", kappa = 2.576, eps = 0.0,verbose = TRUE)


#OR USE THE CODE BELOW FOR OPTIMIZATION OF HYPERPARAMETER

#res0 <- xgb_cv_opt(data = dtrain,
#                   label = is_click,
#                   objectfun = "binary:logistic",
#                   evalmetric = "auc",
#                   n_folds = 5,
#                   acq = "ucb",
#                   init_points = 10,
#                   n_iter = 20)



#BUILD XGBOOST MODEL

xgb_params <- list(booster="gbtree",
                   objective = "binary:logistic",
                   eval_metric = "auc",
                   eta=0.3985,
                   subsample=1.0000,
                   max_depth=22.0000,
                   alpha=1.0000,
                   lambda=0.1224,
                   gamma=0.0000,
                   min_child_weight=3.8733,
                   max_delta_step = 25.1272,
                   colsample_bytree=0.8863)

#nrounds=36.73

bst<-xgboost(params = xgb_params,data=X_train,label =Y_train,nrounds = 36.73)
xgb.importance(feature_names = colnames(X_train), bst) %>% xgb.plot.importance()
xgb.plot.tree(model = bst)



#PREDICTION USING THE MODEL

X_test<-as.matrix(dtest)
test_predict<-predict(bst,X_test)





#WRITE IN SUBMISSION FILE

pred<-0
for(i in 1:length(test_predict)){
  if(test_predict[i]<.07){
    pred[i]=0
  }
  else{
    pred[i]=1
  }
}

submission_data<-data.frame(cbind(test_data$session_id,pred))
colnames(submission_data)<-c("session_id","is_click")
write.table(submission_data,"submission_Tariq.csv",col.names = TRUE,sep = ",",row.names = FALSE)
