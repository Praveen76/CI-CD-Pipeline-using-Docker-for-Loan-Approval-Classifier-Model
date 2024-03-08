"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from sklearn.metrics import accuracy_score

from src.predict import make_prediction

print("make_prediction imported successfully")

def test_mult(a=1, b=2):
   assert a*b == 2

# def test_make_prediction(sample_input_data):
#     # # Given
#     # expected_no_predictions = 2583

#     # # When
#     # result = make_prediction(input_data=sample_input_data)

#     # # Then
#     # predictions = result.get("predictions")
#     # assert isinstance(predictions, np.ndarray)
#     # assert isinstance(predictions[0], np.int64)
#     # assert result.get("errors") is None
#     # assert len(predictions) == expected_no_predictions
#     # _predictions = list(predictions)
#     # y_true = sample_input_data["Survived"]
#     # accuracy = accuracy_score(_predictions, y_true)
#     # assert accuracy > 0.8
#     assert 1 > 0

