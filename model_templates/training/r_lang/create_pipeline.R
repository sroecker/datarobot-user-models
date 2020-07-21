
create_pipeline<-function(target, train_df) {
    # set up the modeling pipeline
    model_recipe <- recipe(target ~ ., data = train_df) %>%
    # Drop constant columns
    step_zv(all_predictors()) %>%
    # Numeric preprocessing
    step_medianimpute(all_numeric()) %>%
    step_normalize(all_numeric(), -all_outcomes()) %>%
    # Categorical preprocessing
    step_other(all_nominal(), -all_outcomes()) %>%
    step_dummy(all_nominal(), -all_outcomes())

    return(model_recipe)
}