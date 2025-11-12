# Air Quality Prediction Model

End-to-end ML service for AQI forecasting, designed for **local** and **AWS** training with clean layering:

- **Ports & Adapters**: abstract storage and trainer; swap local/S3 or XGBoost/other easily.
- **Config-first**: YAML + env override via Pydantic.
- **CLI**: train, evaluate, package.

## Quick start

```bash
python -m pip install -e .
python scripts/train.py --config scripts/local_example.yaml
python scripts/evaluate.py --config scripts/local_example.yaml
python scripts/package_model.py --artifacts out/model --output out/model.tar.gz

# High Level Architecture 

![alt text](images/architecture.png "Overall Architecture")