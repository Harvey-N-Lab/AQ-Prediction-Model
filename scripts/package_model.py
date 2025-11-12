import os
import tarfile
from pathlib import Path
from shutil import copyfile

import typer

app = typer.Typer(help="Package model artifacts to model.tar.gz for SageMaker.")


@app.command()
def main(artifacts: str = "out/model", output: str = "out/model.tar.gz"):
    artifacts_path = Path(artifacts)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure inference.py is included
    infer_src = Path("src/aqi_ml/model/inference.py")
    infer_dst = artifacts_path / "code" / "inference.py"
    infer_dst.parent.mkdir(parents=True, exist_ok=True)
    copyfile(infer_src, infer_dst)

    # Copy schema.json from features.json if present (for online validation)
    feat_path = artifacts_path / "features.json"
    schema_path = artifacts_path / "schema.json"
    if feat_path.exists():
        schema_path.write_text(feat_path.read_text())

    with tarfile.open(output_path, "w:gz") as tar:
        for p in artifacts_path.rglob("*"):
            tar.add(p, arcname=p.relative_to(artifacts_path))
    print(f"Packaged -> {output_path}")


if __name__ == "__main__":
    app()
