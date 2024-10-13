from transformers import Wav2Vec2Processor, HubertForCTC, HubertModel
from datasets import load_dataset
import torch

#dataset = load_dataset("DBD-research-group/BirdSet", "XCM", trust_remote_code=True, cache_dir="B:\DLL\Datasets")
#exit(0)

basemodel = HubertModel.from_pretrained("facebook/hubert-large-ll60k", torch_dtype=torch.float16, attn_implementation=None)
#basemodel = HubertModel.from_pretrained(".\\fairseq\models\hubert_base_ls960_serial", torch_dtype=torch.float16, attn_implementation=None)
#basemodel.load_state_dict(torch.load("fairseq\models\hubert_base_ls960_state_dict.pt"))
load = torch.load(".\\fairseq\models\hubert_large_ll60k_state_dict.pt")
print(load.keys())
basemodel.load_state_dict(load)

print(basemodel)
print()
exit(0)
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
print(processor)
model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
print(model)
#basemodel = Wav2Vec2Model.from_pretrained("facebook/hubert-large-ls960-ft", torch_dtype=torch.float16, attn_implementation=None)
#print(basemodel)
#soundfile? Da dann halt filepath übergeben