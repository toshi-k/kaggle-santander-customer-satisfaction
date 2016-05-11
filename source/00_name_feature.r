
##------------------------------
## read data
##------------------------------

train <- read.csv("../input/train.csv", stringsAsFactors = F)
test <- read.csv("../input/test.csv", stringsAsFactors = F)

##------------------------------
## extract features
##------------------------------

add_train <- data.frame(ID = train$ID)
add_test <- data.frame(ID = test$ID)

## lv.1

add_train <- cbind(add_train, num_zero = rowSums(train[,2:ncol(test)] == 0))
add_test <- cbind(add_test, num_zero = rowSums(test[,2:ncol(test)] == 0))

add_train <- cbind(add_train, num_zero_delta = rowSums(train[,grep("^delta", colnames(train))] == 0))
add_test <- cbind(add_test, num_zero_delta = rowSums(test[,grep("^delta", colnames(train))] == 0))

add_train <- cbind(add_train, num_zero_imp = rowSums(train[,grep("^imp", colnames(train))] == 0))
add_test <- cbind(add_test, num_zero_imp = rowSums(test[,grep("^imp", colnames(train))] == 0))

add_train <- cbind(add_train, num_zero_ind = rowSums(train[,grep("^ind", colnames(train))] == 0))
add_test <- cbind(add_test, num_zero_ind = rowSums(test[,grep("^ind", colnames(train))] == 0))

add_train <- cbind(add_train, num_zero_num = rowSums(train[,grep("^num", colnames(train))] == 0))
add_test <- cbind(add_test, num_zero_num = rowSums(test[,grep("^num", colnames(train))] == 0))

add_train <- cbind(add_train, num_zero_saldo = rowSums(train[,grep("^saldo", colnames(train))] == 0))
add_test <- cbind(add_test, num_zero_saldo = rowSums(test[,grep("^saldo", colnames(train))] == 0))

add_train <- cbind(add_train, num_zero_var = rowSums(train[,grep("^var", colnames(train))] == 0))
add_test <- cbind(add_test, num_zero_var = rowSums(test[,grep("^var", colnames(train))] == 0))

add_train <- cbind(add_train, num_na = rowSums(is.na(train)))
add_test <- cbind(add_test, num_na = rowSums(is.na(test)))

## lv.2

add_train <- cbind(add_train, num_zero_delta_imp = rowSums(train[,grep("^delta_imp", colnames(train))] == 0))
add_test <- cbind(add_test, num_zero_delta_imp = rowSums(test[,grep("^delta_imp", colnames(train))] == 0))

add_train <- cbind(add_train, num_zero_delta_num = rowSums(train[,grep("^delta_num", colnames(train))] == 0))
add_test <- cbind(add_test, num_zero_delta_num = rowSums(test[,grep("^delta_num", colnames(train))] == 0))

add_train <- cbind(add_train, num_zero_imp_aport = rowSums(train[,grep("^imp_aport", colnames(train))] == 0))
add_test <- cbind(add_test, num_zero_imp_aport = rowSums(test[,grep("^imp_aport", colnames(train))] == 0))

add_train <- cbind(add_train, num_zero_imp_op = rowSums(train[,grep("^imp_op", colnames(train))] == 0))
add_test <- cbind(add_test, num_zero_imp_op = rowSums(test[,grep("^imp_op", colnames(train))] == 0))

add_train <- cbind(add_train, num_zero_num_meses = rowSums(train[,grep("^num_meses", colnames(train))] == 0))
add_test <- cbind(add_test, num_zero_num_meses = rowSums(test[,grep("^num_meses", colnames(train))] == 0))

add_train <- cbind(add_train, num_zero_num_op = rowSums(train[,grep("^num_op", colnames(train))] == 0))
add_test <- cbind(add_test, num_zero_num_op = rowSums(test[,grep("^num_op", colnames(train))] == 0))

add_train <- cbind(add_train, num_zero_num_trasp = rowSums(train[,grep("^num_trasp", colnames(train))] == 0))
add_test <- cbind(add_test, num_zero_num_trasp = rowSums(test[,grep("^num_trasp", colnames(train))] == 0))

add_train <- cbind(add_train, num_zero_num_var = rowSums(train[,grep("^num_var", colnames(train))] == 0))
add_test <- cbind(add_test, num_zero_num_var = rowSums(test[,grep("^num_var", colnames(train))] == 0))

add_train <- cbind(add_train, num_zero_saldo_medio = rowSums(train[,grep("^saldo_medio", colnames(train))] == 0))
add_test <- cbind(add_test, num_zero_saldo_medio = rowSums(test[,grep("^saldo_medio", colnames(train))] == 0))

add_train <- cbind(add_train, num_zero_saldo_var = rowSums(train[,grep("^saldo_var", colnames(train))] == 0))
add_test <- cbind(add_test, num_zero_saldo_var = rowSums(test[,grep("^saldo_var", colnames(train))] == 0))

##------------------------------
## save data
##------------------------------

dir.create("../add_features/", showWarnings = FALSE, recursive = TRUE)
write.table(add_train, "../add_features/name_feature_train.csv", sep = ",", row.names = FALSE, quote = FALSE)
write.table(add_test, "../add_features/name_feature_test.csv", sep = ",", row.names = FALSE, quote = FALSE)
