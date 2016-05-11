
data1 <- read.csv("../submission/02_xgb_nummodel10_mvalid0.8424.csv")
data2 <- read.csv("../submission/01_nn_nummodel10_mvalid0.8416.csv")
data3 <- read.csv("../submission/03_xgb_with_knn_nummodel10_mvalid0.8497.csv")

submission <- data1

submission[,2] <- 0.5 * data1[,2] + 0.25 * data2[,2] + 0.25 * data3[,2]

show_df <- data.frame(xgb = data1[,2], nn = data2[,2], knn = data3[,2], sub = submission[,2])
print(head(show_df))

write.table(submission, "../submission/04_all_averaging.csv", row.names = FALSE, sep = ",", quote = FALSE)
