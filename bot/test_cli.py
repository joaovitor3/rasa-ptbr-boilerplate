from argparse import Namespace
from rasa.cli import test
import logging

logger = logging.getLogger(test.__file__)
logger.setLevel('DEBUG')

#Namespace(
#    config=None,
#    cross_validation=False,
#    disable_plotting=False,
#    e2e=True,
#    endpoints='endpoints.yml',
#    evaluate_model_directory=False,
#    fail_on_prediction_errors=False,
#    folds=5,
#    #func=<function test at 0x7fe03bfb4ee0>,
#    loglevel=None,
#    max_stories=None,
#    model='models',
#    nlu='data',
#    no_errors=False,
#    no_warnings=False,
#    out='results/',
#    percentages=[0,
#    25,
#    50,
#    75],
#    runs=3,
#    stories='.',
#    successes=False,
#    url=None
#)

args = Namespace(
        config=None,
        cross_validation=False,
        stories='.',
        endpoints='endpoints.yml',
        out='results/',
        no_errors=False,
        no_warnings=False,
        model='models',
        nlu='data',
        evaluate_model_directory=False,
        fail_on_prediction_errors=False,
        percentages=[0,
        25,
        50,
        75],
        max_stories=None,
        runs=3,
        e2e=True,
        successes=False,
        warnings=True
)
test.run_core_test(args)
test.run_nlu_test(args)