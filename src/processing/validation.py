from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))


from src.config.core import config
from src.processing.data_manager import pre_pipeline_preparation


def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""
    print("input_df imported successfully inside validation file")
    print("input_df columns:", input_df.columns )
    print("input_df shape:", input_df.shape )
    pre_processed = pre_pipeline_preparation(df=input_df)
    validated_data = pre_processed[config.model_config.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class DataInputSchema(BaseModel):
    BorrowerAge: Optional[int]
    CreditScore: Optional[int]
    LoanPurpose: Optional[str]
    LoanType: Optional[str]
    BorrowerTotalMonthlyIncome: Optional[float]
    CLTV: Optional[int]
    DTI: Optional[int]
    ZipCode: Optional[str]
    LeadSourceGroup: Optional[str]
    Cabin: Optional[Union[str, float]]
    BorrowerOwnRent: Optional[str]
    Education: Optional[str]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
    
    