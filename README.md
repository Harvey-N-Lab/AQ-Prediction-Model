# Air Quality Prediction Model

Machine learning service repo containing:
- **Training** (`training/train_xgb.py`): XGBoost regression to predict next-hour PM2.5.
- **Deployment** (`deploy/scripts` + GitHub Actions): Create/Update SageMaker Endpoint.

Provide these GitHub secrets:
- `AWS_IAM_ROLE` (OIDC deploy role ARN)
- `SM_EXEC_ROLE` (SageMaker execution role ARN)
- `MODEL_TAR_S3` (S3 URI to model artifact)

# Overall Architecture 

![alt text](images/architecture.png "Overall Architecture")