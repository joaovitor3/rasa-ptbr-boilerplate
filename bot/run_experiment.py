import mlflow
import os


mlflow.set_experiment('exp')

# Verify problems with permissions
# https://stackoverflow.com/questions/52331254/how-to-store-artifacts-on-a-server-running-mlflow

with mlflow.start_run():
    # The experiment MUST have artifact URI configured
    mlflow.log_artifact('models/20210827-034140.tar.gz')
    print(f'Artifact URI {mlflow.get_artifact_uri()}')