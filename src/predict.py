import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from src import __version__ as _version
from src.config.core import config

print("config imported successfully")

from src.pipeline import lac_pipe
from src.processing.data_manager import load_pipeline
from src.processing.data_manager import pre_pipeline_preparation
from src.processing.validation import validate_inputs

from src.processing.data_manager import load_dataset
# , pre_pipeline_preparation, filter_extreme_vals, data_transformation, data_sanity_check, 


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
lac_pipe= load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data: Union[pd.DataFrame, dict], errors=None) -> dict:
    """Make a prediction using a saved model """

    # validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    
    #validated_data=validated_data.reindex(columns=['Pclass','Sex','Age','Fare', 'Embarked','FamilySize','Has_cabin','Title'])
    validated_data=input_data.reindex(columns=config.model_config.features)
    print("validated_data shape :",validated_data.shape)
    results = {"predictions": None, "version": _version, "errors": errors}
    
    predictions = lac_pipe.predict(validated_data)

    results = {"predictions": predictions, "version": _version, "errors": errors}
    print(results)
    if not errors:

        predictions = lac_pipe.predict(validated_data)
        results = {"predictions": predictions, "version": _version, "errors": errors}
        # print(" prediction data size: ",results.shape)

    return results


if __name__ == "__main__":
    

    data_in = load_dataset(file_name=config.app_config.training_data_file)
    make_prediction(input_data=data_in)
