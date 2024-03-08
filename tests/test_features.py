
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from src.config.core import config
from src.processing.features import MissValImputer

print("========================= config imported successfully=========================")

def test_add(a=1, b=2):
   assert a+b == 3
   
# def test_miss_val_imputer(sample_input_data):
#     # Given
#     print(" sample_input_data columns:", sample_input_data.columns)
#     # transformer = MissValImputer(
#     #     variables=config.model_config.num_vars, 
#     # )
#     # assert np.isnan(sample_input_data.loc[3416, 'DTI'])

#     # # When
#     # subject = transformer.fit(sample_input_data).transform(sample_input_data)

#     # # Then
#     # assert subject.loc[3416,'DTI'] == 25.0
#     assert 1 > 0
    
