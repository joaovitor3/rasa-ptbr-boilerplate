import mlflow
import yaml
from test_cli import run_tests
import os.path


def file_exists(filename):
    return os.path.isfile(filename) 

def yaml_to_dict(filename: str):
    with open(filename, "r") as yaml_file:
        try:
            yaml_dict = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(exc)
    return yaml_dict

def log_artifacts(model_filepath: str, config_filepath: str, domain_filepath: str):
    mlflow.log_artifact(model_filepath)
    mlflow.log_artifact(config_filepath)
    mlflow.log_artifact(domain_filepath)

def log_metrics(failed_stories_filepath: str):
    failed_stories = 0
    if file_exists(failed_stories_filepath):
        failed_stories_dict = yaml_to_dict(failed_stories_filepath)
        failed_stories = len(failed_stories_dict["stories"])

    mlflow.log_metric("failed_stories", failed_stories)


if __name__ == "__main__":
    domain_dict = yaml_to_dict("domain.yml")
    entities = domain_dict["entities"]
    intents = domain_dict["intents"]

    config_dict = yaml_to_dict("config.yml")
    pipeline = config_dict["pipeline"]
    policies = config_dict["policies"]

    model_filepath = 'models/20210827-034140.tar.gz'
    model_params = {
        "pipeline": pipeline,
        "policies": policies
    }

    config_filepath = 'config.yml'
    domain_filepath = "domain.yml"
    nlu_filepath = "data/nlu.yml"
    rules_filepath = "data/rules.yml"
    stories_filepath = "data/stories.yml"

    # run_tests()
    mlflow.set_experiment('exp')
    with mlflow.start_run():
        log_artifacts(model_filepath, config_filepath, domain_filepath)
        log_metrics("results/failed_test_stories.yml")

    
