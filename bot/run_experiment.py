import mlflow
import yaml
from test_cli import run_tests
import os.path

import pandas as pd

from sklearn.metrics import classification_report

from rasa.cli.utils import get_validated_path
from rasa.model import get_model, get_model_subdirectories
from rasa.nlu.model import Interpreter
import rasa.shared.nlu.training_data.loading as nlu_loading

def load_interpreter(model_path):
  """
  This loads the Rasa NLU interpreter. It is able to apply all NLU
  pipeline steps to a text that you provide it.
  """
  model = get_validated_path(model_path, "model")
  model_path = get_model(model)
  _, nlu_model = get_model_subdirectories(model_path)
  return Interpreter.load(nlu_model)




def add_predictions(dataf, nlu):
    """This function will add prediction columns based on `nlu`"""
    pred_blob = [nlu.parse(t)['intent'] for t in dataf['text']]
    return (dataf
            [['text', 'intent']]
            .assign(pred_intent=[p['name'] for p in pred_blob])
            .assign(pred_confidence=[p['confidence'] for p in pred_blob]))

def file_exists(filename):
    return os.path.isfile(filename) 

def yaml_to_dict(filename: str):
    with open(filename, "r") as yaml_file:
        try:
            yaml_dict = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(exc)
    return yaml_dict

def log_artifacts(artifacts):
    for artifact in artifacts.values():
        mlflow.log_artifact(artifact)

def log_metrics(failed_stories_filepath: str, nlu_metrics: dict):
    failed_stories = 0
    if file_exists(failed_stories_filepath):
        failed_stories_dict = yaml_to_dict(failed_stories_filepath)
        failed_stories = len(failed_stories_dict["stories"])

    mlflow.log_metric("failed_stories", failed_stories)

    for metric_name, metric_value in nlu_metrics.items():
        mlflow.log_metric(metric_name, metric_value)


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

    artifacts = {
        'config': 'config.yml',
        'domain': "domain.yml",
        'nlu': "data/nlu.yml",
        'rules': "data/rules.yml",
        'stories': "data/stories.yml"
    }

    # run_tests()

    # Source: https://rasa.com/blog/evaluating-rasa-nlu-models-in-jupyter/
    # First make a list of dictionaries that contain the utterances.
    nlu_interpreter = load_interpreter("models/20210827-034140.tar.gz")
    train_data = nlu_loading.load_data("data/nlu.yml")
    data_list = [m.as_dict() for m in train_data.intent_examples]

    # Turn this list into a dataframe and add predictions using the nlu-interpreter
    df_intents = pd.DataFrame(data_list).pipe(add_predictions, nlu=nlu_interpreter)

    df_summary = (df_intents
                .groupby('pred_intent')
                .agg(n=('pred_confidence', 'size'),
                    mean_conf=('pred_confidence', 'mean')))

    report = classification_report(y_true=df_intents['intent'], y_pred=df_intents['pred_intent'], output_dict=True)

    nlu_metrics = {
        'accuracy': report['accuracy'],
        'precision': report['macro avg']['precision'],
        'recall': report['macro avg']['recall'],
        'f1-score': report['macro avg']['f1-score']
    }

    mlflow.set_experiment('exp')
    with mlflow.start_run():
        log_artifacts(artifacts)
        log_metrics("results/failed_test_stories.yml", nlu_metrics)

