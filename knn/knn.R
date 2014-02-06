digits_knn <- function(num_neighbors, algorithm) {
  
  # Check argument validity
  # TODO: check for valid num_neighbors
  if (!(algorithm %in% c("kd_tree", "cover_tree", "brute"))) {
    stop("invalid algorithm")
  }
  
  # Read the data
  #train <- read.csv("../train.csv")
  #test <- read.csv("../test.csv")
  #labels <- as.factor(train[,1])
  #train <- train[,-1]
  
  # todo (for future applications): ensure train and test data are same length
  
  # Do the KNN
  set.seed(1)
  library(FNN)
  result <- knn(train, test, labels, num_neighbors, FALSE, algorithm)
  
  # Format and write output (and time how long it takes)
  ids <- 1:length(result)
  t0 <- proc.time()
  output <- data.frame(ids, result)
  t <- proc.time() - t0
  print(t)
  
  colnames(output) <- c("ImageId", "Label")
  filename <- sprintf("result_knn_%s_%s_extended.csv", num_neighbors, algorithm)
  write.table(output, filename, quote=FALSE, sep=",", row.names=FALSE) 
  
  return(t)

}
