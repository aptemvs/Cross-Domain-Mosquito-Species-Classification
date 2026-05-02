from collections.abc import Iterator
import itertools
from copy import deepcopy

from model.experiment_config import ExperimentConfig
from model.trial_config import TrialConfig


def generate_trials(config: ExperimentConfig) -> Iterator[TrialConfig]:
    dump = config.model_dump()
    iterable_keys = filter(lambda key: isinstance(dump[key], list), dump.keys())
    iterable_values = dump.fromkeys(iterable_keys)

    for values in itertools.product(*iterable_values):
        trial_dump = deepcopy(dump)
        trial_dump.update(zip(iterable_keys, values))

        trial = TrialConfig.model_validate(trial_dump)

        yield trial
