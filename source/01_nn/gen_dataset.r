
##------------------------------
## library
##------------------------------

library(xgboost)

##------------------------------
## initialize
##------------------------------

result_all <- NULL

set.seed(1000)

##------------------------------
## read data
##------------------------------

## main table ----------

train <- read.csv("../../input/train.csv", stringsAsFactors = F)
test <- read.csv("../../input/test.csv", stringsAsFactors = F)

## name features ----------

train_name_features <- read.csv("../../add_features/name_feature_train.csv", stringsAsFactors = F)
test_name_features <- read.csv("../../add_features/name_feature_test.csv", stringsAsFactors = F)

train <- merge(train, train_name_features, by = "ID")
test <- merge(test, test_name_features, by = "ID")

## add ID as feature ----------

train$ID_feature <- train$ID
test$ID_feature <- test$ID

## fill na ----------

train[is.na(train)] <- -999999
test[is.na(test)] <- -999999

y <- as.numeric(train$train_target)

train_y <- train$train_target
train_x <- train
train_x$train_target <- NULL

dir.create("../../input/", showWarnings = FALSE, recursive = TRUE)
write.table(train_x, "../../input/nn_train_x.csv", sep = ",", row.names = FALSE)
write.table(train_y, "../../input/nn_train_y.csv", sep = ",", row.names = FALSE)
write.table(test, "../../input/nn_test_x.csv", sep = ",", row.names = FALSE)
