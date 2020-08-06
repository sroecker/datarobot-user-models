from ngboost import NGBClassifier, NGBRegressor

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import pandas as pd


def create_regression_model() -> NGBRegressor:
    """
    Create a regression model.

    Returns
    -------
    NGBRegressor
        NGBoost regressor model
    """
    ng_reg = NGBRegressor(random_state=42)
    return ng_reg


def create_binary_classification_model() -> NGBClassifier:
    """
    Create a binary classification model.

    Returns
    -------
    NGBClassifier
        NGBoost classifier model
    """
    ng_clf = NGBClassifier(seed=123)
    return ng_clf


def make_classifier_pipeline(X: pd.DataFrame) -> Pipeline:
    """
    Make the classifier pipeline with the required preprocessor steps and estimator in the end.

    Parameters
    ----------
    X
        X containing all the required features for training

    Returns
    -------
    Pipeline
        Classifier pipeline with preprocessor and estimator
    """
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    num_features = list(X.select_dtypes(include=numerics).columns)

    # This example model only uses numeric features and drops the rest
    num_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_features)])

    # create model
    estimator = create_binary_classification_model()

    # pipeline with preprocessor and estimator bundled
    classifier_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])
    return classifier_pipeline


def make_regressor_pipeline(X: pd.DataFrame) -> Pipeline:
    """
    Make the regressor pipeline with the required preprocessor steps and estimator in the end.

    Parameters
    ----------
    X
        X containing all the required features for training

    Returns
    -------
    Pipeline
        Regressor pipeline with preprocessor and estimator
    """
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    num_features = list(X.select_dtypes(include=numerics).columns)

    # This example model only uses numeric features and drops the rest
    num_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="mean")), ("standardize", StandardScaler())]
    )
    preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_features)])

    # create model
    estimator = create_regression_model()

    # pipeline with preprocessor and estimator bundled
    regressor_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])
    return regressor_pipeline
