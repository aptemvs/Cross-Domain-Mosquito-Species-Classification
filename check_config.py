from framework.config import load_config

# loads and generates JSON schema. throws if incorrect json
config = load_config("configs/default_experiment.json")
print(config)
