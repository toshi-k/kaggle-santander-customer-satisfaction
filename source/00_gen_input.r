
##------------------------------
## Read data
##------------------------------

cat("==> Read data\n")
train <- read.csv("../dataset/train.csv", stringsAsFactors = FALSE)
test <- read.csv("../dataset/test.csv", stringsAsFactors = FALSE)

train_target <- train$TARGET
train$TARGET <- NULL

data_all <- rbind(train, test)

## Remove constant cols -----
cat("==> Remove colstant cols\n")

remove <- c()
for(name in names(data_all)){
	if (class(data_all[[name]]) == "integer") {
		u <- unique(data_all[[name]])
		if (length(u) == 1){
			data_all[[name]] <- NULL
		} 
	}
}

## Remove overlap cols -----
cat("==> Remove overlap cols\n")

train_names <- names(data_all)
remove <- c()
for(i in 1:length(train_names)){
	if(i != length(train_names)){
		for (k in (i+1):length(train_names)){
			if(identical(data_all[,i], data_all[,k]) == TRUE){
				remove <- c(remove, k)
			}
		}
	}
}
data_all <- data_all[,-remove]

## Replace NA -----
cat("==> Replace NA\n")

data_all[data_all == -999999] <- NA
data_all[data_all == 9999999999] <- NA

## Split Data -----
cat("==> Split Data\n")

train2 <- data_all[1:nrow(train),]
test2 <- data_all[(nrow(train)+1):nrow(data_all),]
train2 <- cbind(train2, train_target)

train2_unique <- train2[!duplicated(train2[,-1]),]
test2_unique <- test2[!duplicated(test2[,-1]),]

##------------------------------
## Save data
##------------------------------

cat("==> Save Data\n")
dir.create("../input/", showWarnings = FALSE, recursive = TRUE)

write.table(train2, "../input/train.csv", sep=",", row.names = FALSE)
write.table(test2, "../input/test.csv", sep=",", row.names = FALSE)
write.table(train2_unique, "../input/train_unique.csv", sep=",", row.names = FALSE)
write.table(test2_unique, "../input/test_unique.csv", sep=",", row.names = FALSE)
