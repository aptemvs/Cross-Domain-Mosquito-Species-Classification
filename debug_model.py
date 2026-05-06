import torch
from framework.config import load_config
from framework.loss.loss import loss_all
from framework.model_output import ModelOutput
from framework.utilization import build_model, choose_device

config = load_config('configs/default_experiment.json')
device = choose_device("auto")
model = build_model(config, device)
model.eval()

batch_size = 8
frames = 200
features = torch.randn(batch_size, frames, config["n_mels"], device=device)
lengths = torch.tensor([200, 190, 180, 170, 160, 150, 140, 130], device=device)

with torch.no_grad():
    outputs: ModelOutput = model(features, lengths)
    species_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device=device)
    domain_labels = torch.tensor([0, 1, 0, 1, 2, 2, 3, 3], device=device)
    total_loss, loss_items = loss_all(outputs, species_labels, domain_labels)

print(model)

print("device", device)
print("model_backend", config["model_backend"])
print("species_logits", tuple(outputs.species_logits.shape))
print("domain_logits", tuple(outputs.domain_logits.shape))
print("embeddings", tuple(outputs.embeddings.shape))
print("total_loss", float(total_loss))
print("loss_items", loss_items)
