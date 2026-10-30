from importlib.metadata import version

import pipecat_guardrails


def test_version_matches_installed_metadata():
    assert pipecat_guardrails.__version__ == version("pipecat-guardrails")


def test_pipecat_classifier_interface_is_available():
    # Layers 2 and 3 (classifier and LLM judge) are built on this interface.
    from pipecat.classifiers.base_classifier import BaseClassifier, YesNoQuestion

    assert hasattr(BaseClassifier, "yes_no")
    assert YesNoQuestion(instructions="Is this a test?").instructions == "Is this a test?"
