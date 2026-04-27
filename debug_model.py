import torch
from framework.config import load_config
from framework.utilization import build_model, choose_device

config = load_config('configs/default_experiment.json')
device = choose_device("auto")
model = build_model(config, device)
model.eval()

features = torch.randn(2, 200, config['n_mels'], device=device)
lengths = torch.tensor([200, 160], device=device)

with torch.no_grad():
    outputs = model(features, lengths)

print(model)

print('device', device)
print('model_backend', config['model_backend'])
print('species_logits', tuple(outputs['species_logits'].shape))
print('domain_logits', tuple(outputs['domain_logits'].shape))
