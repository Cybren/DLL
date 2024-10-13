#from fairseq import fairseq
import torch

ckpt_path = "fairseq\models\hubert_base_ls960.pt"
#models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])

#model = models[0]

d = torch.load(ckpt_path)

print(d.keys())
print(d["args"])
print(d["model"])
