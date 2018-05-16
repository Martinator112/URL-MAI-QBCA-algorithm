filename <- './gaussian_external_adjusted_mutual_info_index_results.csv'
filename2 <- './gaussian_external_rand_score_index_results.csv'
filename3 <- './gaussian_external_normalized_mutual_info_index_results.csv'

adjusted_mutual_info_index <- read.csv(filename)
rands_score_index <- read.csv(filename2)
normalized_mutual_info_index <- read.csv(filename3)

summary(adjusted_mutual_info_index)
summary(rands_score_index)
summary(normalized_mutual_info_index)

adjusted_mutual_info_index <- adjusted_mutual_info_index[-c(5)]
rands_score_index <- rands_score_index[-c(5)]
normalized_mutual_info_index <- normalized_mutual_info_index[-c(5)]

boxplot(adjusted_mutual_info_index, ylab="Score", main="Adjusted mutual info index")
boxplot(rands_score_index, ylab="Score", main="Rand score index")
boxplot(normalized_mutual_info_index, ylab="Score", main="Normalized mutual info index")