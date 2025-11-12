from aqi_ml.ports import RegistryPort

class NoopRegistry(RegistryPort):
    def register(self, model_artifact_dir: str, metrics: dict) -> None:
        # Extend later for SageMaker Model Registry or MLflow
        pass
