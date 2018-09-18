# Data Science Specialization: Practical Machine Learning, Final Project
## Date: 2018-09-12
### R Version: 3.5.1
### RStudio Version: 1.1.456

# Clear Workspace

rm(list = ls())

# Load Libraries

if(!require(dplyr)){install.packages("dplyr")}
if(!require(readr)){install.packages("readr")}
if(!require(caret)){install.packages("caret")}
if(!require(rpart)){install.packages("rpart")}
if(!require(rattle)){install.packages("rattle")}
if(!require(stringr)){install.packages("stringr")}
if(!require(ggplot2)){install.packages("ggplot2")}
if(!require(lubridate)){install.packages("lubridate")}
if(!require(randomForest)){install.packages("randomForest")}

library(dplyr)
library(readr)
library(caret)
library(rpart)
library(rattle)
library(stringr)
library(ggplot2)
library(lubridate)
library(randomForest)

# Load Datasets

url_train <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url_test <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

all <- read_csv(file = url_train)
validation <- read_csv(file = url_test)

rm(url_train, url_test)

# Exploratory Analysis

glimpse(training)
table(training$user_name)
summary(training)
head(training)

# Data Slicing: Creating Training & Testing Sets

  ## Due to the small size of the testing set (n = 20), we'll consider this the "validation" set
  ## We can split the training set to create a training and test set with 75% and 25%, resp.

set.seed(716)

in_train <- createDataPartition(y = all$classe, p = .75, list = FALSE)

training <- all[in_train, ]
testing <- all[-in_train, ]

rm(in_train, all)

# ADDITIONAL PREPROCESSING

## Removing Feature of Interest

# training <- select(training, -classe)

    ## Yo, hold up until models built.

## Removing near-zero variability features

training <- training[, -nearZeroVar(x = training)] # Remove features with NZV (Near-Zero Variance)

    ### These variables have no or little variability and would not make useful features in ML model

## Principal Components Analysis (PCA)

non_char <- training[, which(sapply(training, class) != "character")]
non_char <- as.data.frame(sapply(non_char, as.numeric))
corr_mat <- abs(cor(non_char))

findCorrelation(corr_mat, cutoff = 0.75)

    ### This doesn't work, too many NA values

rm(corr_mat, non_char)

## Removing high NA count (Remove if 95% or more are NA values)

training <- training[, which(colSums(is.na(training)) / nrow(training) < .95)]

    ## Note: There is some information loss, despite *extremely high threshold* - this ensures robust imputation is impossible

## Removing First 6 Columns

training <- training[, -c(1:6)]

# CROSS-VALIDATION

cv_folds <- trainControl(method = "cv", number = 7) # K-folds cross-validation, 14 folds of equal size for ~1,000-instance folds

    ## Tradeoffs: We've chosen a larger 'k' than default (10), which means less bias introduced, but more variance
      ### Hence more accuracy and variability; no replacement (unlike bootstrapping)
      ### Much more computationally expensive [Change in write-up]

# MODEL: Decision Trees

dt_fit <- train(form = classe ~ ., data = training, method = "rpart", na.action = "na.omit", trControl = cv_folds)

print(dt_fit)
print(dt_fit$finalModel)

fancyRpartPlot(dt_fit$finalModel)

dt_pred <- predict(object = dt_fit, newdata = testing)
dt_conmat <- confusionMatrix(as.factor(testing$classe), dt_pred)

plot(dt_conmat$table, main = "Decision Tree Confusion Matrix", color = "skyblue")

print(dt_conmat)

# MODEL: Random Forest

rf_fit <- randomForest(as.factor(classe) ~ ., data = training, na.action = "na.omit")

rf_pred <- predict(object = rf_fit, newdata = testing)
rf_conmat <- confusionMatrix(as.factor(testing$classe), rf_pred)

plot(rf_conmat$table, main = "Random Forest Confusion Matrix", color = "lightgreen")

print(rf_conmat)

# MODEL: Boosting

bs_fit <- train(form = as.factor(classe) ~ ., data = training, method = "gbm", na.action = "na.omit", trControl = cv_folds)

bs_pred <- predict(object = bs_fit, newdata = testing)
bs_conmat <- confusionMatrix(as.factor(testing$classe), bs_pred)

plot(bs_conmat$table, main = "Boosted Regression Confusion Matrix", color = "tomato")

print(bs_conmat)

plot(bs_fit)

# FINAL MODEL SELECTION

accuracy <- data.frame("Decision Tree" = dt_conmat$overall[1:4],
                       "Random Forest" = rf_conmat$overall[1:4],
                       "Generalized Boosted Regression" = bs_conmat$overall[1:4],
                       check.names = FALSE)



