from typing import Any, List, Optional

from pydantic import BaseModel
from src.processing.validation import DataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[int]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "LeadSourceGroup": "Internet",
                        "ZipCode": 78,
                        "LoanNumber": 41031910948,
                        "LoanPurpose": "Refinance",
                        "LoanType": "VA",
                        "TotalLoanAmount": 414000.0,
                        "CreditScore": 760,
                        "CLTV": 89.03,
                        "DTI": 17,
                        "BorrowerAge": 51,
                        "BorrowerTotalMonthlyIncome": 0.0,
                        "BorrowerOwnRent": "Own",
                        "Approved": 0,
                        "TotalIncome": 0.0,
                        "Education": "UnderGrad"
                    }
                ]
            }
        }
