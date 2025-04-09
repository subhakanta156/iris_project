import mlflow
from mlflow.tracking import MlflowClient

# Experiment and model name — same as used during training
experiment_name = "Iris_RF_Experiment"
model_name = "IrisRandomForestModel"

# Init the MLflow client
client = MlflowClient()

# Getting experiment details (will return None if it doesn't exist)
experiment = client.get_experiment_by_name(experiment_name)

if experiment is None:
    print(f"Bruh, experiment '{experiment_name}' not found 😶")
    exit()

# Get the latest run from this experiment (most recent one)
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],  # sort by latest
    max_results=1
)

if not runs:
    print("No runs found, did you train the model yet? 😅")
    exit()

# Grab run ID and model location
run_id = runs[0].info.run_id
model_uri = f"runs:/{run_id}/model"

# Registering model — it’ll show up in MLflow UI under "Models" tab
result = mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)

print(f" Model registered as '{model_name}' (version: {result.version})")