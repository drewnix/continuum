# scripts/deploy.py
import os
import boto3  # Example: for AWS S3 upload

def upload_model_to_s3(filepath, bucket_name, object_name=None):
    if object_name is None:
        object_name = os.path.basename(filepath)
    
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(filepath, bucket_name, object_name)
        print(f"Model uploaded to S3 bucket {bucket_name} as {object_name}")
    except Exception as e:
        print(f"Error uploading model: {e}")

if __name__ == "__main__":
    filepath = 'models/logistic_regression_model.pkl'
    bucket_name = 'your-s3-bucket-name'
    upload_model_to_s3(filepath, bucket_name)