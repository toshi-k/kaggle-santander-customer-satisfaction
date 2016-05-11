
prj_name <- "01_nn"
candi <- list.files(paste0("../submission/", prj_name, "/"))
print(candi)

scores <- as.numeric(substring(candi, nchar(candi)-9, nchar(candi)-4))

info <- data.frame(filename = candi, score = scores)

num_model <- 10

for(i in 1:num_model){
	target_file <- as.character(info$filename[order(info$score, decreasing = TRUE)[i]])
	print(target_file)
	datai <- read.csv(paste0("../submission/", prj_name, "/", target_file))
	if(i==1){
		data <- datai
	}else{
		data$TARGET <- data$TARGET + datai$TARGET
	}
}

data$TARGET <- data$TARGET / num_model

meanvalid <- mean(sort(info$score, decreasing = TRUE)[1:num_model])
write.table(data, paste0("../submission/", prj_name, "_nummodel", num_model, "_mvalid", sprintf("%.4f", meanvalid), ".csv"), row.names = FALSE, sep=",", quote = FALSE)
