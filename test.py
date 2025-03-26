import boto3

# Replace with your desired bucket name and AWS region
bucket_name = "my-python-s3-bucket-12345"
region = "us-east-1"  # Change as needed

# Initialize S3 client
s3_client = boto3.client('s3')

# Create S3 bucket
try:
    s3_client.create_bucket(
        Bucket=bucket_name,
        CreateBucketConfiguration={'LocationConstraint': region}
    )
    print(f"Bucket '{bucket_name}' created successfully in {region}!")
except Exception as e:
    print(f"Error creating bucket: {e}")