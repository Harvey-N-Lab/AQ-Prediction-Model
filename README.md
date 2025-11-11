# Air Quality Prediction Model

Glue ETL and model training templates for AQI prediction.

- `glue/etl_job.py`: Transform raw OpenAQ + OpenWeather to Parquet with lag/rolling features.
- `training/train_xgb.py`: Local/SageMaker-compatible XGBoost training script.
- `pipelines/` and `registry/`: placeholders for SageMaker AMT + Model Registry automation.

# Overall Architecture 

![alt text](images/architecture.png "Overall Architecture")