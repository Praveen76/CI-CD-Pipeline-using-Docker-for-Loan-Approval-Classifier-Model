import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

import sys
from pathlib import Path


file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))


import typing as t
import re


from src import __version__ as _version
from src.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config



import pandas as pd

def data_sanity_check(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy of the DataFrame
    checked_df = df.copy()
    print("===========data_sanity_check started for df===========")
    print("checked_df columns: ", checked_df.columns)
    print("checked_df shape: ", checked_df.shape)
    # Remove duplicate records.
    checked_df.drop_duplicates(subset='LoanNumber', keep='first', inplace=True)

    # Filter out wrong BorrowerYearsInSchool records
    checked_df = checked_df[checked_df['BorrowerYearsInSchool'] >= 1]

    # Strip off white space
    checked_df.LoanPurpose = [entry.strip() for entry in checked_df.LoanPurpose]
    checked_df.LoanType = [entry.strip() for entry in checked_df.LoanType]

    checked_df.reset_index(drop=True, inplace=True)
    print("=========================== data_sanity_check execution done ===========================")
    
    return checked_df



def data_transformation(df: pd.DataFrame):
    # Create Approved column
    df['Approved'] = [1 if not pd.isna(date) else 0 for date in df['ApprovalDate']]

    # Recategorize Loan Purpose column
    LP = ['Refinance', 'Purchase']
    df['LoanPurpose'] = [v if v in LP else 'Refinance' for v in df['LoanPurpose']]

    # Calculate difference between current date and DateAdded in days
    df['DateAdded'] = pd.to_datetime(df['DateAdded'])
    df['Diff'] = (pd.Timestamp.now().normalize() - df['DateAdded']).dt.days

    df['Approved'] = [0 if ((x >= 90) and (y == 'Purchase')) else 1 for (x, y) in zip(df['Diff'], df['LoanPurpose'])]
    df['Approved'] = [0 if ((x >= 45) and (y == 'Refinance')) else 1 for (x, y) in zip(df['Diff'], df['LoanPurpose'])]

    # Convert to category type
    df['Approved'] = df['Approved'].astype('category')
    df['IsCoBorrowerower'] = df['IsCoBorrowerower'].astype('category')

    # Drop 'Diff' column
    df.drop('Diff', axis=1, inplace=True)

    # Update 'BorrowerOwnRent' column
    OwnRent = ['Own', 'Rent']
    df['BorrowerOwnRent'] = [v if v in OwnRent else 'Own' for v in df['BorrowerOwnRent']]
    df['BorrowerOwnRent'].fillna(df['BorrowerOwnRent'].mode().values[0], inplace=True)

    # Add Borrower & Co-Borrower's income
    df['TotalIncome'] = df['BorrowerTotalMonthlyIncome'] + df['CoBorrowerTotalMonthlyIncome']

    # Regroup LeadSourceGroup
    LSR = ['Internet', 'TV', 'Radio', 'Repeat Client']
    df['LeadSourceGroup'] = [v if v in LSR else 'Other' for v in df['LeadSourceGroup']]

    # Update 'ZipCode' column
    zips = ['75', '76', '77', '78', '79']
    df['ZipCode'] = [str(zp)[:2] if str(zp)[:2] in zips else 'Other' for zp in df['ZipCode']]

    # Create bins for Education
    bins = [0, 12, 16, 18, df['BorrowerYearsInSchool'].max()]
    group = ['Higher School', 'UnderGrad', 'PostGrad', 'PHD']
    df['Education'] = pd.cut(df['BorrowerYearsInSchool'], bins, labels=group)

    print("=========================== data_transformation execution done ===========================")
    return df

def filter_extreme_vals(df: pd.DataFrame):
    ## Make a copy of the DataFrame
    filtered_df = df.copy()

    ## Filter out extreme outliers
    filtered_df = filtered_df[filtered_df.CLTV < 110]
    filtered_df = filtered_df[filtered_df.TotalLoanAmount < 600000]
    filtered_df = filtered_df[filtered_df.CreditScore > 550]
    filtered_df = filtered_df[filtered_df.TotalIncome <= 30000]

    print("=========================== filter_extreme_vals execution done ===========================")
    
    return filtered_df


def pre_pipeline_preparation(*, df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy of the DataFrame
    processed_df = df.copy()
    print("================== pre_pipeline_preparation started for variables==================")
    print("processed_df columns: ", processed_df.columns)
    print("processed_df shape: ", processed_df.shape)
    processed_df = data_sanity_check(processed_df)   # Run data sanity checks
    print("================== data_sanity_check done for variables==================")
    processed_df = data_transformation(processed_df)   # Perform preliminary data transformation
    print("================== data_sanity_check done for variables==================")
    processed_df = data_transformation(processed_df)   # Filter extreme values
    print("================== filter_extreme_vals done for variables==================")
    processed_df.drop(labels=config.model_config.unused_fields, axis=1, inplace=True)   # Drop unnecessary variables

    print("=========================== pre_pipeline_preparation execution done ===========================")
    return processed_df



def _load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe


def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    print("========================= Data has been imported successfully =========================")
    print("df columns :", dataframe.columns)
    transformed = pre_pipeline_preparation(df=dataframe)
    print("========================= Data has been transformed successfully =========================")
    return transformed


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
