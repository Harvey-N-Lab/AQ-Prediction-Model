"""
Lightweight SageMaker Training Job launcher that reuses the same YAML config.

Note: This is a minimal stub so you can iterate quickly.
Replace with a proper AMT launcher later when you want tuning.
"""

import argparse
import boto3
import json
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to YAML config")
parser.add_argument("--image_uri", required=True, help="Training image URI (e.g., XGBoost)")
parser.add_argument("--role_arn", required=True, help="SageMaker execution role ARN")
parser.add_argument("--input_s3", required=True, help="s3://bucket/prefix containing training data")
parser.add_argument("--output_s3", required=True, help="s3://bucket/prefix for model output")
parser.add_argument("--instance_type", default="ml.m5.xlarge")
parser.add_argument("--instance_count", type=int, default=1)
args = parser.parse_args()

sm = boto3.client("sagemaker")
job_name = "aqi-xgb-" + str(int(time.time()))

# We pass the YAML content via hyperparameter "user_config" (your container must read it)
with open(args.config, "r") as f:
    cfg_text = f.read()

resp = sm.create_training_job(
    TrainingJobName=job_name,
    AlgorithmSpecification={"TrainingImage": args.image_uri, "TrainingInputMode": "File"},
    RoleArn=args.role_arn,
    InputDataConfig=[
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": os.path.join(args.input_s3, "train/"),
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
        },
        {
            "ChannelName": "val",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": os.path.join(args.input_s3, "val/"),
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
        },
    ],
    OutputDataConfig={"S3OutputPath": args.output_s3},
    ResourceConfig={
        "InstanceType": args.instance_type,
        "InstanceCount": args.instance_count,
        "VolumeSizeInGB": 50,
    },
    StoppingCondition={"MaxRuntimeInSeconds": 3600},
    HyperParameters={"user_config": cfg_text},
)

print(json.dumps({"training_job_name": job_name}, indent=2))
