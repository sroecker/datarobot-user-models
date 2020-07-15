import logging
import os
import pandas as pd

from datarobot_drum.drum.common import LOGGER_NAME_PREFIX
from pathlib import Path
from mlpiper.components.connectable_component import ConnectableComponent

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
except ImportError:
    error_message = (
        "rpy2 package is not installed."
        "Install datarobot-drum using 'pip install datarobot-drum[R]'"
        "Available for Python>=3.6"
    )
    logger.error(error_message)
    exit(1)


pandas2ri.activate()
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
R_FIT_PATH = os.path.join(CUR_DIR, "fit.R")

r_handler = ro.r


class RFit(ConnectableComponent):
    def __init__(self, engine):
        super(RFit, self).__init__(engine)
        self.target_name = None
        self.output_dir = None
        self.estimator = None
        self.positive_class_label = None
        self.negative_class_label = None
        self.custom_model_path = None
        self.input_filename = None
        self.weights = None
        self.weights_filename = None
        self.target_filename = None
        self.num_rows = None

    def configure(self, params):
        super(RFit, self).configure(params)
        self.custom_model_path = self._params["__custom_model_path__"]
        self.input_filename = self._params["inputFilename"]
        self.target_name = self._params.get("targetColumn")
        self.output_dir = self._params["outputDir"]
        self.positive_class_label = self._params.get("positiveClassLabel")
        self.negative_class_label = self._params.get("negativeClassLabel")
        self.weights = self._params["weights"]
        self.weights_filename = self._params["weightsFilename"]
        self.target_filename = self._params.get("targetFilename")
        self.num_rows = self._params["numRows"]

        r_handler.source(R_FIT_PATH)
        r_handler.init(self.custom_model_path)

    def _materialize(self, parent_data_objs, user_data):
        X, y, class_order, row_weights = shared_preprocessing(self)

        df = X
        df.iloc[:, self.target_name] = y
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(X)
            r_row_weights = ro.conversion.py2rpy(row_weights)

        r_handler.outer_fit(r_df,
                            output_dir=self.output_dir,
                            class_order=class_order,
                            row_weights=r_row_weights
                            )

        make_sure_artifact_is_small(self.output_dir)
        return []


# TODO: move these guys
def make_sure_artifact_is_small(output_dir):
    """
    # TODO: docstring
    :param output_dir:
    :return:
    """
    MEGABYTE = 1024 * 1024
    GIGABYTE = 1024 * MEGABYTE
    root_directory = Path(output_dir)
    dir_size = sum(f.stat().st_size for f in root_directory.glob("**/*") if f.is_file())
    logger.info("Artifact directory has been filled to {} Megabytes".format(dir_size / MEGABYTE))
    assert dir_size < 10 * GIGABYTE


def shared_preprocessing(fit_class):
    """
    # TODO: docstring, lots of comments
    :param fit_class:
    :return:
    """
    df = pd.read_csv(fit_class.input_filename)
    if fit_class.num_rows == "ALL":
        fit_class.num_rows = len(df)
    else:
        fit_class.num_rows = int(fit_class.num_rows)
    if fit_class.target_filename:
        X = df.sample(fit_class.num_rows, random_state=1)
        y = pd.read_csv(fit_class.target_filename, index_col=False).sample(
            fit_class.num_rows, random_state=1
        )
        assert len(y.columns) == 1
        assert len(X) == len(y)
        y = y.iloc[:, 0]
    else:
        X = df.drop(fit_class.target_name, axis=1).sample(
            fit_class.num_rows, random_state=1, replace=True
        )
        y = df[fit_class.target_name].sample(fit_class.num_rows, random_state=1, replace=True)

    if fit_class.weights_filename:
        row_weights = pd.read_csv(fit_class.weights_filename).sample(
            fit_class.num_rows, random_state=1, replace=True
        )
    elif fit_class.weights:
        if fit_class.weights not in X.columns:
            raise ValueError(
                "The column name {} is not one of the columns in "
                "your training data".format(fit_class.weights)
            )
        row_weights = X[fit_class.weights]
    else:
        row_weights = None

    class_order = (
        [fit_class.negative_class_label, fit_class.positive_class_label]
        if fit_class.negative_class_label
        else None
    )

    return X, y, class_order, row_weights


