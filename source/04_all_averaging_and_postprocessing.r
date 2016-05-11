
data1 <- read.csv("../submission/02_xgb_nummodel10_mvalid0.8424.csv")
data2 <- read.csv("../submission/01_nn_nummodel10_mvalid0.8416.csv")
data3 <- read.csv("../submission/03_xgb_with_knn_nummodel10_mvalid0.8497.csv")

submission <- data1

preds <- 0.5 * data1[,2] + 0.25 * data2[,2] + 0.25 * data3[,2]

tc <- read.csv("../dataset/test.csv")

## This postprocessing is imspired by ZFTurbo's script
## https://www.kaggle.com/zfturbo/santander-customer-satisfaction/to-the-top-v3

nv <- tc['num_var33'] + tc['saldo_medio_var33_ult3'] + tc['saldo_medio_var44_hace2'] + tc['saldo_medio_var44_hace3'] + tc['saldo_medio_var33_ult1'] + tc['saldo_medio_var44_ult1']

preds[nv > 0] = 0
preds[tc['var15'] < 23] = 0
preds[tc['saldo_medio_var5_hace2'] > 160000] = 0
preds[tc['saldo_var33'] > 0] = 0
preds[tc['var38'] > 3988596] = 0
preds[tc['var21'] > 7500] = 0
preds[tc['num_var30'] > 9] = 0
preds[tc['num_var13_0'] > 6] = 0
preds[tc['num_var33_0'] > 0] = 0
preds[tc['imp_ent_var16_ult1'] > 51003] = 0
preds[tc['imp_op_var39_comer_ult3'] > 13184] = 0
preds[tc['saldo_medio_var5_ult3'] > 108251] = 0

submission[,2] <- preds

write.table(submission, "../submission/04_all_averaging_and_postprocessig.csv", row.names = FALSE, sep = ",", quote = FALSE)
