#!/usr/bin/env bash
set -euo pipefail
REGION=${REGION:-ap-southeast-1}
MODEL_NAME=${MODEL_NAME:?}
MODEL_DATA_URL=${MODEL_DATA_URL:?}   # s3://.../model.tar.gz or model artifact
IMAGE_URI=${IMAGE_URI:?}             # e.g., XGBoost built-in image
ENDPOINT_NAME=${ENDPOINT_NAME:-aqi-xgb-endpoint}
EXEC_ROLE_ARN=${EXEC_ROLE_ARN:?}

aws sagemaker create-model --region "$REGION"   --model-name "$MODEL_NAME"   --primary-container Image="$IMAGE_URI",ModelDataUrl="$MODEL_DATA_URL"   --execution-role-arn "$EXEC_ROLE_ARN" || true

aws sagemaker create-endpoint-config --region "$REGION"   --endpoint-config-name "${ENDPOINT_NAME}-cfg"   --production-variants VariantName=AllTraffic,ModelName="$MODEL_NAME",InitialInstanceCount=1,InstanceType=ml.m5.large || true

aws sagemaker create-endpoint --region "$REGION"   --endpoint-name "$ENDPOINT_NAME"   --endpoint-config-name "${ENDPOINT_NAME}-cfg" || aws sagemaker update-endpoint --region "$REGION"   --endpoint-name "$ENDPOINT_NAME"   --endpoint-config-name "${ENDPOINT_NAME}-cfg"
