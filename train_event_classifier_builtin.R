# Run locally once. Keep this script outside the deployed Shiny Rmd.
library(dplyr)
library(caret)
library(quanteda)
library(quanteda.textmodels)

# 1. Read data and create the same Event_Group labels as the Shiny project
raw <- read.csv('January2015toNovember2025.csv',
                stringsAsFactors = FALSE, colClasses = 'character')

event_data <- raw |>
  mutate(
    Event_Division = substr(Event, 1, 1),
    Event_Group = case_when(
      Event_Division == '1' ~ 'Violence / Persons / Animals',
      Event_Division == '2' ~ 'Transportation Incidents',
      Event_Division == '3' ~ 'Fires and Explosions',
      Event_Division == '4' ~ 'Falls, Slips, Trips',
      Event_Division == '5' ~ 'Harmful Substances / Environments',
      Event_Division == '6' ~ 'Contact with Objects / Equipment',
      Event_Division == '7' ~ 'Overexertion / Bodily Reaction',
      Event_Division == '9' ~ 'Nonclassifiable / Other',
      TRUE ~ 'Unknown'
    )
  ) |>
  filter(!is.na(Final.Narrative), nzchar(trimws(Final.Narrative)),
         Event_Group != 'Unknown') |>
  select(Final.Narrative, Event_Group)
rm(raw)
gc(verbose = FALSE)

print(table(event_data$Event_Group))

# 2. Stratified train/test split (same style as the GLM project)
set.seed(123)
index <- createDataPartition(event_data$Event_Group, p = 0.8, list = FALSE)
event_train <- event_data[index, ]
event_test <- event_data[-index, ]

# 3. Reusable text preprocessing (identical for training and prediction)
make_tokens <- function(x) {
  z <- tokens(x, remove_punct = TRUE, remove_numbers = TRUE)
  z <- tokens_tolower(z)
  z <- tokens_remove(z, stopwords('en'))
  tokens_wordstem(z, language = 'english')
}

train_dfm <- dfm(make_tokens(event_train$Final.Narrative))
train_dfm <- dfm_trim(train_dfm, min_docfreq = 5, docfreq_type = 'count')
event_features <- names(topfeatures(train_dfm, 5000))
train_dfm <- dfm_match(train_dfm, features = event_features)

# 4. Fit built-in multinomial Naive Bayes (like glm(), but for text)
event_model <- textmodel_nb(
  train_dfm,
  y = factor(event_train$Event_Group),
  distribution = 'multinomial'
)

# 5. Held-out test (vocabulary learned only from training data)
test_dfm <- dfm(make_tokens(event_test$Final.Narrative))
test_dfm <- dfm_match(test_dfm, features = event_features)
event_pred <- as.character(predict(event_model, newdata = test_dfm))
event_pred[ntoken(test_dfm) == 0] <- 'Unknown'
truth <- event_test$Event_Group

accuracy <- mean(event_pred == truth)
classes <- sort(unique(event_train$Event_Group))
f1 <- vapply(classes, function(k) {
  tp <- sum(event_pred == k & truth == k)
  fp <- sum(event_pred == k & truth != k)
  fn <- sum(event_pred != k & truth == k)
  if (2 * tp + fp + fn == 0) return(NA_real_)
  2 * tp / (2 * tp + fp + fn)
}, numeric(1))
metrics <- list(
  accuracy = accuracy,
  macro_f1 = mean(f1, na.rm = TRUE),
  per_class_f1 = f1,
  n_train = nrow(event_train),
  n_test = nrow(event_test),
  confusion = table(Actual = truth, Predicted = event_pred)
)

# 6. Save trained model only; Shiny will not run training
classifier <- list(model = event_model, features = event_features)
saveRDS(classifier, 'event_classifier.rds', compress = 'xz')
saveRDS(metrics, 'event_classifier_metrics.rds')
cat(sprintf('Saved! Accuracy: %.3f; Macro F1: %.3f\n', accuracy, metrics$macro_f1))
cat(sprintf('Model file: %.2f MiB\n',
            file.info('event_classifier.rds')$size / 1024^2))
