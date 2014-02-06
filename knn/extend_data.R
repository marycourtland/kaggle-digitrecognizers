## This takes the digit data and finds sum of differences between adjacent pixels
## When finding difference between 2 pixels, only consider:
## (a) two pixels vertically adjacent
## (b) two pixels horizontally adjacent

## Two new columns will be added to the data frame:
## (a) Sum of horizontal differences
## (b) Sum of vertical differences

## NB: take the absolute value of the differences, so that the pixel differences
## on either side of a stroke (in the digit) don't cancel each other out.

extend_data_soads <- function(data, img_size = 28) {
  # These will be the new columns
  horz_sums <- numeric(nrow(data))
  vert_sums <- numeric(nrow(data))
  
  ## Loop through each row of the data
  ## Compute horizontal and vertical differences and put them in the new cols
  
  for (r in 1:nrow(data)) {
    datum <- data[r,]
    horz_sums[r] <- diff_sum_horizontal(datum, img_size)
    vert_sums[r] <- diff_sum_vertical(datum, img_size)
  }
  
  # Add the new columns to the data frame
  data[,length(data)+1] <- horz_sums
  data[,length(data)+1] <- vert_sums
  
  return(data)
}

diff_sum_horizontal <- function(datum, img_size = 28) {
  ## Sum the differences of adjacent horizontal pixels in the single datum
  sum <- 0
  for (row_num in 1:img_size) { # Loop through each row of pixels
    for (px_num in 1:(img_size - 1)) { # Loop through each pixel in that row (except the last pixel)
      index <- (row_num - 1) * img_size + px_num
      sum <- sum + abs(datum[, index] - datum[, index + 1])
    }
  }
  return(sum)
}

diff_sum_vertical <- function(datum, img_size) {
  ## Sum the differences of adjacent vertical pixels in the single datum
  sum <- 0
  for (col_num in 1:img_size) { # Loop through each column of pixels
    for (px_num in 1:(img_size - 1)) { # Loop through each pixel in that column (except the last pixel)
      index <- (px_num - 1) * img_size + col_num
      sum <- sum + abs(datum[, index] - datum[, index + img_size])
    }
  }
  return(sum)
}


####################################################
# Here are three test cases of 3x3 images
# (Sums were computed by hand)
# 
# Test image 1: (a sortof vertical stroke)
# 0.0  0.5  0.0
# 0.0  0.8  0.1
# 0.0  0.4  0.0
# sum_horz = 3.3
# sum_vert = 0.9
# 
# 
# Test image 2: (a sortof horizontal stroke)
# 0.2  0.1  0.0
# 0.9  1.0  0.7
# 0.0  0.0  0.1
# 
# sum_horz = 0.7
# sum_vert = 4.8
# 
# 
# Test image 3: (a sortof diagonal stroke)
# 1.0  0.8  0.0
# 0.1  1.0  0.7
# 0.0  0.0  0.9
# 
# sum_horz = 3.1
# sum_vert = 3.1
# 
# Compiled data with all three test images:
# 0.0  0.5  0.0  0.0  0.8  0.1  0.0  0.4  0.0
# 0.2  0.1  0.0  0.9  1.0  0.7  0.0  0.0  0.1
# 1.0  0.8  0.0  0.1  1.0  0.7  0.0  0.0  0.9
# 
# Compiled data with all three test images AND sums:
# 0.0  0.5  0.0  0.0  0.8  0.1  0.0  0.4  0.0  3.3  0.9
# 0.2  0.1  0.0  0.9  1.0  0.7  0.0  0.0  0.1  0.7  4.8
# 1.0  0.8  0.0  0.1  1.0  0.7  0.0  0.0  0.9  3.1  3.1
#
# To run the test, call:
#   extend_data(get_test_data(), 3)
#
# The resulting dataframe should be the same as the above table.

get_test_data <- function() {
  data <- data.frame(
    c(0.0, 0.2, 1.0),
    c(0.5, 0.1, 0.8),
    c(0.0, 0.0, 0.0),
    c(0.0, 0.9, 0.1),
    c(0.8, 1.0, 1.0),
    c(0.1, 0.7, 0.7),
    c(0.0, 0.0, 0.0),
    c(0.4, 0.0, 0.0),
    c(0.0, 0.1, 0.9)
  )
  colnames(data) <- c("11", "12", "13", "21", "22", "33", "31", "32", "33")
  return(data)
}
