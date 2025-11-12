import typer
from aqi_ml.training.pipeline import TrainingPipeline
from aqi_ml.config import AppConfig

app = typer.Typer(help="Train AQI model locally or on AWS.")

@app.command()
def main(config: str):
    cfg = AppConfig.from_yaml(config)
    TrainingPipeline(cfg).run()

if __name__ == "__main__":
    app()
