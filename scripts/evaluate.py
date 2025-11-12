import json, typer
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from math import sqrt

app = typer.Typer(help="Evaluate a packaged model on a dataset (simple baseline script).")

@app.command()
def main(predictions: str = "out/model/val_predictions.csv"):
    p = Path(predictions)
    if not p.exists():
        typer.echo(f"Not found: {predictions}")
        raise typer.Exit(1)
    df = pd.read_csv(p)
    y = df["y_true"].values
    yhat = df["y_pred"].values

    rmse = sqrt(mean_squared_error(y, yhat))
    mape = mean_absolute_percentage_error(y, yhat)
    metrics = {"rmse": rmse, "mape": mape}
    typer.echo(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    app()
