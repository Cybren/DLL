# Deep Learning Lab SS2024 Repo SSL-2
Repo for exploring if HuBERT can be utilized as an SSL model for increasing birdcall classification performance.
Birdset and Fairseq folders contain only the respective repos, needed for HuBERT code and testing on Birdset.
The requirements_*.txt are for generating a conda env for fairseq and birdset respectivly.
The birdset env is used for everything except HuBERT pretraining.
Pretraining was tried on XCM and XCL the IES-Cluster with the slurm scripts in ./slurm but was unsuccessfull due to errors in the preporcessing of the wave files.
Specifically the computation of the MFCCs from the wav-files failed at one point, making pretraining impossible.
In finetuneHubert and finetuneHubertMulticlass the finetuning of Hubert on the birdcall classification task was performed.
For this, pretrained HuBERT models from Huggingface trained on LibriSpeech were used.
In finetuneHubert the Model was finetuned using the multilabel setting while the finetuneHubertMulticlass used the multiclass setting.
Since the finetuning on the multilabel-task was unsuccessfull multiclass was tried as well to see if that task was solvable.
The features produced by HuBERT on the birdcalls were very large ([512, 249]) the linear probing was extended by a CNN inbetween the model and the linear layer to transform the features.
The model and metrics used for finetuning are in Model.py.
To evaluate why the finetuning did not work the testModel and compareFeautres notebooks were used to look at the outputs of finetuned models and the features produced by HuBERT on birdcalls and human speech (from LibriSpeech).
Results from experiments are in lightning_logs and can be viewed with tensorboard --logdir lightning_logs.