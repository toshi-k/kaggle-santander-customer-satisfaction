
##------------------------------
## library
##------------------------------

library(xgboost)

##------------------------------
## initialize
##------------------------------

result_all <- NULL

set.seed(1000)

for(i in 1:300){

	cat("\ncount: ", i, "-----\n")

	##------------------------------
	## params
	##------------------------------

	myparam_colsample_bytree <- runif(1, 0.5, 0.99)
	myparam_subsample <- runif(1, 0.7, 0.99)
	myparam_eta <- runif(1, 0.005, 0.05)
	myparam_max_depth <- sample(4:8, 1)
	myparam_max_delta_step <- sample(0:3, 1)
	myparam_base_score <- runif(1, 0.05, 0.5)

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

	## fill na ----------

	train[is.na(train)] <- -999999
	test[is.na(test)] <- -999999

	#Building the model
	set.seed(1000 + i)
	param <- list(objective = "binary:logistic",
					booster = "gbtree",
					eval_metric = "auc",
					colsample_bytree = myparam_colsample_bytree,
					subsample = myparam_subsample,
					eta = myparam_eta,
					max.depth = myparam_max_depth,
					max_delta_step = myparam_max_delta_step,
					base_score = myparam_base_score)

	y <- as.numeric(train$train_target)

	train_y <- train$train_target
	train_x <- train
	train_x$train_target <- NULL

	test_id <- test$ID

	train_x$ID <- NULL
	test$ID <- NULL

	dtrain <- xgb.DMatrix(data.matrix(train_x), label = train_y)

	model_cv <- xgb.cv(param = param,
						data = dtrain,
						nrounds = 100000,
						nfold = 10,
						early.stop.round = 100,
						print.every.n = 100L)

	best_score <- max(model_cv$test.auc.mean)
	cat("\nbest_score: ", best_score, "\n")
	best_itr <- which.max(model_cv$test.auc.mean)

	result_new <- data.frame(ID = i,
		myparam_colsample_bytree = myparam_colsample_bytree,
		myparam_subsample = myparam_subsample,
		myparam_eta = myparam_eta,
		myparam_max_depth = myparam_max_depth,
		myparam_max_delta_step = myparam_max_delta_step,
		myparam_base_score = myparam_base_score,
		best_itr = as.integer(best_itr * 1.1),
		best_score = best_score)

	print(result_new)

	prj_name <- "02_xgb"

	result_all <- rbind(result_all, result_new)
	write.table(result_all, "result_all.csv", sep = ",", row.names = FALSE)

	model <- xgb.train(data = dtrain, params = param, nrounds = best_itr)

	imp <- xgb.importance(feature_names = colnames(train_x), model = model)

	dtest <- xgb.DMatrix(data.matrix(test))

	## predict -----
	submission <- predict(model, dtest)
	submission <- data.frame(ID = test_id, TARGET = submission)

	dir.create(paste0("../../submission/", prj_name, "/"), showWarnings = FALSE, recursive = TRUE)
	filename <- paste0("../../submission/", prj_name, "/", prj_name, "_itr", i, "_valid", sprintf("%.4f", best_score), ".csv")
	write.table(submission, filename, sep = ",", row.names = FALSE)
}
