# Packages ----------------------------------------------------------------
library(tidyverse)
library(tidymodels)
theme_set(theme_light())

# Explore data ------------------------------------------------------------
ikea <- readr::read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-11-03/ikea.csv')
glimpse(ikea)


ikea %>%
  mutate(price_dkk = price * 1.63) %>%
  pivot_longer(depth:width, names_to = "dim") %>%
  ggplot(aes(value, price_dkk, color = dim)) +
  geom_point(alpha = 0.4, show.legend = FALSE) + 
  facet_wrap(~dim, scales = "free_x") + 
  labs(x = "Size (cm)", y = "Price (DKK)", title = "Width seems to have the biggest impact on price") + 
  scale_y_log10() 

ikea %>%
  ggplot(aes(x = fct_rev(fct_infreq(category)), fill = category)) +
  geom_bar(show.legend = FALSE) + 
  labs(x = "Category", y = "Count") + 
  coord_flip() 

ikea_df <- ikea %>%
  mutate(price_dkk = log10(price * 1.63)) %>%
  select(price_dkk, category, depth, height, width) %>%
  mutate_if(is.character, as.factor)
  
# Build models ------------------------------------------------------------
set.seed(123)
ikea_split <- initial_split(ikea_df, strata = price_dkk)
ikea_train <- training(ikea_split)
ikea_test <- testing(ikea_split)

set.seed(234)
ikea_folds <- vfold_cv(ikea_train, v = 10)

lr_mod <- linear_reg() %>%
  set_engine("lm")

lr_recipe <- recipe(price_dkk~ ., data = ikea_train) %>%
  step_other(category, threshold = 0.05) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_zv(all_predictors()) %>%
  step_knnimpute(depth, height, width)

lr_workflow <- workflow() %>%
  add_model(lr_mod) %>%
  add_recipe(lr_recipe)


library(usemodels)
use_ranger(price_dkk ~ ., data = ikea_train)

ranger_recipe <- 
  recipe(price_dkk ~ ., data = ikea_train) %>%
  step_other(category, threshold = 0.05) %>%
  step_knnimpute(depth, height, width)

ranger_spec <- 
  rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>% 
  set_mode("regression") %>% 
  set_engine("ranger") 

ranger_workflow <- 
  workflow() %>% 
  add_recipe(ranger_recipe) %>% 
  add_model(ranger_spec) 

set.seed(42761)
ranger_tune <-
  tune_grid(ranger_workflow, resamples = ikea_folds, grid = 10)

# Explore results ---------------------------------------------------------
show_best(ranger_tune, metric = "rmse")
show_best(ranger_tune, metric = "rsq")

autoplot(ranger_tune)

final_rf <- ranger_workflow %>%
  finalize_workflow(select_best(ranger_tune, metric = "rmse"))

# Comparing LR vs RF on the test data

ikea_lr_fit <- last_fit(lr_workflow, ikea_split)
ikea_lr_fit %>% 
  collect_metrics()

ikea_rf_fit <- last_fit(final_rf, ikea_split)
ikea_rf_fit %>% 
  collect_metrics()
# Random Forest model is superior!

collect_predictions(ikea_rf_fit) %>%
  ggplot(aes(price_dkk, .pred)) +
  geom_abline(lty = 2, color = "gray50", size = 2) +
  geom_point(alpha = 0.5, color = "midnightblue") +
  coord_fixed()

library(vip)

imp_spec <- ranger_spec %>%
  finalize_model(select_best(ranger_tune, metric = "rmse")) %>%
  set_engine("ranger", importance = "permutation")

workflow() %>%
  add_recipe(ranger_recipe) %>%
  add_model(imp_spec) %>%
  fit(ikea_train) %>%
  pull_workflow_fit() %>%
  vip(aesthetics = list(alpha = 0.8, fill = "midnightblue"))




  
  
