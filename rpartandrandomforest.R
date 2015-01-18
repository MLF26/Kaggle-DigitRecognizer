setwd("~/Centrale/Machine Learning/Digit Recognizer")

train <- read.csv("~/Centrale/Machine Learning/Digit Recognizer/train.csv")
test <- read.csv("~/Centrale/Machine Learning/Digit Recognizer/test.csv")
train$label <- factor(train$label)

# Split in train and validation sets
split =  sample(seq(1,42000), 29400)
train1 = train[split,] 
train2 = train[-split,] # validation set

# Cart classification tree
library(rpart)
cartTree = rpart(label ~ ., data=train1, method="class")
cartPredictions = predict(cartTree, newdata=train2, type="class")
confusionMatrix <- table(train2$label, cartPredictions) # Table creates the confusion matrix
hitRate <- sum(diag(confusionMatrix))/nrow(train2) # Calculation of the hit rate
print(paste("Hit Rate :", as.character(floor(hitRate*100)),"%")) # Print the hit rate
predictions = predict(cartTree, newdata=test, type = "class")
submit = data.frame(ImageId = seq(1, 28000), label = predictions)
write.csv(submit, file = "rpart.csv", row.names = FALSE)

# Random forest
library(randomForest)
randomForest = randomForest(label ~ ., data=train1, nodesize=10, ntree=50, do.trace=TRUE)
randomForestPredictions = predict(randomForest, newdata=train2)
confusionMatrix = table(train2$label, randomForestPredictions) # Table creates the confusion matrix
hitRate <- sum(diag(confusionMatrix))/nrow(train2) # Calculation of the hit rate
print(paste("Hit Rate :", as.character(floor(hitRate*100)),"%")) # Print the hit rate
predictions = predict(randomForest, newdata=test)
submit = data.frame(ImageId = seq(1, 28000), label = predictions)
write.csv(submit, file = "randomforest.csv", row.names = FALSE)