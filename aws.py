import json
import logging
import pandas as pd
import numpy as np
import requests
from io import StringIO
import boto3
import os
import re
from dateutil import parser
from datetime import datetime
import uuid
from botocore.exceptions import BotoCoreError, NoCredentialsError
from boto3.session import Session

# Configure logging
logging.basicConfig(level=logging.INFO)

# ============================
# Configure Logging for Lambda
# ============================
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# ============================
# Pandas Display Options
# ============================
THRESHOLD_MISSING_VALUES = 50
THRESHOLD_COLUMN_UNIQUE = 20

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)

# ============================
# Utility Functions
# ============================


def log_dataframe_info(df, message):
    """Logs DataFrame shape, data types, and missing values."""
    logging.info(f"\n=== {message} ===")
    logging.info(f"Shape: {df.shape}")
    logging.info(f"Data Types:\n{df.dtypes}")
    logging.info(f"Missing Values:\n{df.isnull().sum()}")


def process_large_file_in_chunks(input_file_path, output_file_path, chunk_size=10000):
    """Processes a large file in chunks to avoid memory overflow."""
    logging.info(f"Processing file in chunks: {input_file_path}")

    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    chunk_iter = pd.read_csv(input_file_path, chunksize=chunk_size)

    for i, chunk in enumerate(chunk_iter):
        logging.info(f"Processing chunk {i + 1}")
        cleaned_chunk = clean_and_fill_dataset(chunk)
        cleaned_chunk.to_csv(output_file_path, mode='a', index=False,
                             header=not os.path.exists(output_file_path))
        logging.info(f"Chunk {i + 1} written to file")

    logging.info(f"All chunks processed and saved to {output_file_path}")
    print("-done process_large_file_in_chunks-")


def drop_duplicate_rows(df):
    log_dataframe_info(df, "Before Dropping Duplicates")
    df.drop_duplicates(inplace=True)
    log_dataframe_info(df, "After Dropping Duplicates")
    print("--done drop_duplicate_rows--")
    return df


def clean_high_missing_columns(df, threshold=THRESHOLD_MISSING_VALUES):
    log_dataframe_info(df, "Before Cleaning High Missing Columns")
    df.replace(["None", "none", "NULL", "null", "NA",
               "nan", ""], np.nan, inplace=True)

    for col in df.columns:
        if df[col].isnull().sum() > threshold:
            logging.warning(
                f"Dropping Column: {col} (Too Many Missing Values)")
            df.drop(columns=[col], inplace=True)

    log_dataframe_info(df, "After Cleaning High Missing Columns")
    print("---done clean_high_missing_columns---")
    return df


def remove_currency_symbols(df):
    currency_pattern = r'[\$\€\£\¥\₹\₽\₩\฿\₺\%\(\)\#\!\*\@]'
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = (df[col].astype(str)
                       .apply(lambda x: re.sub(currency_pattern, '', x).strip())
                       .str.replace(',', '')
                       .replace(['-', '', ' '], np.nan))
            try:
                df[col] = df[col].astype(float)
            except ValueError:
                pass
    print("----remove_currency_symbols----")
    return df


def drop_symbol_only_columns(df):
    symbol_pattern = r'^[^\w\d]+$'
    symbol_columns = [col for col in df.columns if df[col].dropna().astype(
        str).str.match(symbol_pattern).all()]
    df = df.drop(columns=symbol_columns)
    if symbol_columns:
        logging.info(f"Dropped columns with only symbols: {symbol_columns}")
    print("-----drop_symbol_only_columns-----")
    return df


def convert_columns_to_datetime64(df):
    """Converts date-like columns to datetime64 format."""
    for column in df.columns:
        if should_convert_column(column, df[column]):
            original_dtype = df[column].dtype
            converted = df[column].apply(safe_to_datetime)

            if converted.notna().sum() > 0:
                df[column] = converted
            else:
                df[column] = df[column].astype(original_dtype)
    print("------convert_columns_to_datetime64------")
    return df


def should_convert_column(column, series):
    if series.dtype.name not in ['object', 'string']:
        return False

    if 'date' in column.lower() or 'time' in column.lower():
        return True

    sample = series.dropna().sample(min(20, len(series.dropna())), random_state=42)

    date_like_count = sum(is_date_like(v) for v in sample)

    print("-------should_convert_column-------")
    return date_like_count / len(sample) > 0.7


def is_date_like(value):
    """Checks if a value looks like a date."""
    if pd.isna(value) or isinstance(value, (bool, int, float)):
        return False

    value = str(value).strip()


    DATE_PATTERNS = [
         r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$",                      # YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD
         r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\s+\d{1,2}:\d{1,2}:\d{1,2}$",  # YYYY-MM-DD HH:MM:SS
         r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{4}$",                      # DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY
         r"^\d{4}\d{2}\d{2}$",                                    # YYYYMMDD
         r"^\d{2}[-/.]\d{2}[-/.]\d{2}$",                          # DD-MM-YY
         r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{2}$",                      # D-M-YY
         r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{4}\s+\d{1,2}:\d{1,2}$",    # DD-MM-YYYY HH:MM
         r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{4}\s+\d{1,2}:\d{1,2}:\d{1,2}$",  # DD-MM-YYYY HH:MM:SS
         r"^\d{2}[-/.]\d{4}$",                                    # MM-YYYY or MM.YYYY
    ]

    for pattern in DATE_PATTERNS:
        if re.match(pattern, value):
            print("done is data like true")
            return True
    print("done is data like flase")
    return False


def safe_to_datetime(value):
    """Converts values safely to datetime."""
    if is_date_like(value):
        try:
            return pd.to_datetime(value, errors='coerce', infer_datetime_format=True)
        except Exception:
            return pd.NaT
    print("*done safe to datetime*")
    return value


def clean_and_fill_dataset(df):
    """Cleans and fills missing values with appropriate logic."""
    logging.info("\n===== STARTING DATA CLEANING PROCESS =====")
    df = drop_duplicate_rows(df)
    df = clean_high_missing_columns(df)
    df = convert_columns_to_datetime64(df)
    df = fill_missing_values(df)
    df = drop_symbol_only_columns(df)
    log_dataframe_info(df, "FINAL CLEANED DATA")
    print("**done clean and fill dataset**")
    return df


def fill_missing_values(df):
    """Fills NaN values based on the column type."""
    log_dataframe_info(df, "Before Filling Missing Values")
    df = remove_currency_symbols(df)

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            dtype = detect_column_type(df[col])

            try:
                if dtype == "category" or df[col].dtype == "object":
                    fill_value = df[col].mode(
                    )[0] if not df[col].mode().empty else "Unknown"
                    df[col].fillna(fill_value, inplace=True)
                    df[col] = df[col].astype("category")

                elif dtype == "bool":
                    df[col].fillna(False, inplace=True)
                    df[col] = df[col].astype(bool)

                elif dtype == "int64":
                    median_value = df[col].median()
                    df[col].fillna(int(median_value) if pd.notnull(
                        median_value) else 0, inplace=True)

                elif dtype == "float64":
                    median_value = df[col].median()
                    df[col].fillna(median_value if pd.notnull(
                        median_value) else 0.0, inplace=True)

            except Exception as e:
                logging.error(f"Error processing column '{col}': {e}")

    log_dataframe_info(df, "After Filling Missing Values")
    print("***fill_missing_values***")
    return df


def detect_column_type(series):
    if series.dropna().isin([True, False, "TRUE", "FALSE", "True", "False", 1, 0]).all():
        return "bool"

    numeric_check = pd.to_numeric(series.dropna(), errors="coerce")
    if numeric_check.notna().all():
        print("done detect_column_type int64 or float64")
        return "int64" if numeric_check.astype(float).mod(1).sum() == 0 else "float64"
    print("done detect_column_type category or object")
    return "category" if series.nunique() < THRESHOLD_COLUMN_UNIQUE else "object"


def fill_missing_values_at_the_end(df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing values (NaN) using enhanced logic at the end."""
    df = df.replace("nan", np.nan)

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            unique_count = df[col].nunique(dropna=True)

            if pd.api.types.is_numeric_dtype(df[col]):
                if unique_count > 20:
                    fill_value = df[col].median()
                    df[col].fillna(fill_value, inplace=True)
                else:
                    fill_value = df[col].mode(
                    )[0] if not df[col].mode().empty else 0
                    df[col].fillna(fill_value, inplace=True)
            else:
                fill_value = df[col].mode(
                )[0] if not df[col].mode().empty else "Unknown"
                df[col].fillna(fill_value, inplace=True)

                if df[col].dtype == "object" or df[col].dtype.name == "category":
                    df[col] = df[col].astype("category")

    df = convert_object_to_category(df)
    print("****done fill_missing_values_at_the_end****")
    return df


def convert_object_to_category(df):
    """Converts all object columns to category."""
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")
    print("*****done convert_object_to_category*****")
    return df


def convert_dataframe_to_json(df):
    """Converts DataFrame to JSON with appropriate key-value pairs."""
    try:
        json_data = df.to_dict(orient="records")
        print("done convert_dataframe_to_json")
        return json_data
    except Exception as e:
        logging.error(f"Error converting DataFrame to JSON: {e}")
        return []


def upload_file_to_s3(file_path, bucket_name, s3_key):
    """Uploads a file to the specified S3 bucket."""
    try:
        s3_client = boto3.client('s3')
        s3_client.upload_file(file_path, bucket_name, s3_key)
        logging.info(f"File uploaded to S3: s3://{bucket_name}/{s3_key}")
        print("=upload file to s3=")
        return f"s3://{bucket_name}/{s3_key}"
    except Exception as e:
        logging.error(f"Failed to upload file to S3: {e}")
        raise

## corrected processed_file_key and processed_file_url without callback url
# version 2

def lambda_handler(event, context=None):
    logging.info("======== Entered Lambda Function ========")

    try:
        # ===========================
        # Validate Required Fields
        # ===========================
        required_fields = ["original_file_url", "original_file_key", "callback_url"]
        missing_fields = [field for field in required_fields if field not in event]

        if missing_fields:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": f"Missing required fields: {', '.join(missing_fields)}"})
            }

        # Extract Required Fields
        original_file_url = event["original_file_url"]
        original_file_key = event["original_file_key"]
        callback_url = event["callback_url"]

        # Optional AWS Credentials and Bucket Name
        bucket_name = event.get("bucket_name", "your-default-bucket")
        aws_access_key = event.get("aws_access_key")
        aws_secret_key = event.get("aws_secret_key")

        logging.info(f"Extracted - Processed File URL: {original_file_url}, Original File Key: {original_file_key}, Callback URL: {callback_url}")
        logging.info(f"Using bucket: {bucket_name}")

        # ===========================
        # Extract Original File Name
        # ===========================
        original_file_name = os.path.basename(original_file_url.split("?")[0])

        if not original_file_name.endswith(".csv"):
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Invalid file format. Only CSV files are supported."})
            }

        logging.info(f"Original file name extracted: {original_file_name}")

        # Generate Unique Filename
        unique_id = str(uuid.uuid4())#[:8]
        unique_file_name = f"{unique_id}_cleaned_{original_file_name}"
        s3_key = f"processed_data/{unique_file_name}"
        logging.info(f"Generated unique file name: {unique_file_name}, S3 Key: {s3_key}")

        # ===========================
        # Fetch File from URL
        # ===========================
        try:
            response = requests.get(original_file_url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            logging.error(f"Failed to fetch file: {str(e)}")
            return {
                "statusCode": 500,
                "body": json.dumps({"error": f"Failed to fetch file: {str(e)}"})
            }

        logging.info("File successfully fetched from URL")

        # ===========================
        # Load File into DataFrame
        # ===========================
        file_content = response.content.decode("utf-8")
        df = pd.read_csv(StringIO(file_content))
        logging.info(f"Loaded CSV with shape: {df.shape}")

        # Perform Data Cleaning
        cleaned_df = df.drop_duplicates()
        logging.info(f"Cleaned data shape: {cleaned_df.shape}")

        # Save to Temporary File
        temp_path = f"/tmp/{unique_file_name}"
        cleaned_df.to_csv(temp_path, index=False)
        logging.info(f"Cleaned file saved at: {temp_path}")

        # ===========================
        # Setup S3 Client
        # ===========================
        try:
            if aws_access_key and aws_secret_key:
                session = Session(aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)
                s3_client = session.client("s3")
            else:
                s3_client = boto3.client("s3")

            # Upload to S3
            s3_client.upload_file(temp_path, bucket_name, s3_key)
            logging.info(f"File uploaded to S3 at: {s3_key}")

            # Generate Presigned URL
            presigned_url = s3_client.generate_presigned_url("get_object", Params={"Bucket": bucket_name, "Key": s3_key}, ExpiresIn=86400)

        except (BotoCoreError, NoCredentialsError) as e:
            logging.error(f"S3 Upload Failed: {str(e)}")
            return {
                "statusCode": 500,
                "body": json.dumps({"error": f"S3 Upload Failed: {str(e)}"})
            }

        # Cleanup Temporary File
        os.remove(temp_path)
        logging.info(f"Temporary file {temp_path} deleted.")

        # ===========================
        # Prepare Response
        # ===========================
        response_payload = {
            "original_file_key": original_file_key, # key provided by backend api
            "processed_file_url": presigned_url, # presigned url generated after uploading cleaned file to S3
            "processed_file_key": s3_key, # Unique file name generated for cleaned file while uploading to S3
            "callback_url": callback_url
        }
        logging.info(f"Payload to Next Lambda: {response_payload}")

        return {
            "statusCode": 200,
            "body": json.dumps(response_payload)
        }

    except Exception as e:
        logging.error(f"Lambda Execution Error: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Internal server error: {str(e)}"})
        }