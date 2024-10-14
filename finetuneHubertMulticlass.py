from typing import Any


if(__name__ == "__main__"):
    from birdset.datamodule import DatasetConfig
    from birdset.datamodule.birdset_datamodule import BirdSetDataModule
    from birdset.datamodule.components.transforms import BirdSetTransformsWrapper
    from transformers import Wav2Vec2Model, HubertModel, Trainer #TODO schauen wegen HubertModel
    import torch
    from torchmetrics import Accuracy, Precision
    import time
    import fairseq
    from tqdm import tqdm
    import lightning as lt
    from huggingface_hub import PyTorchModelHubMixin
    from lightning.pytorch.loggers import TensorBoardLogger
    from torchmetrics.classification.average_precision import AveragePrecision
    from torchmetrics.classification.auroc import AUROC
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from sklearn.metrics import average_precision_score
    import time
    from Model import Modelwrapper

    task = "multiclass"
    dataset_name = "HSN"
            
    model_name = "facebook/hubert-xlarge-ll60k"
    basemodel = Wav2Vec2Model.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation=None)
    #ckpt_path_small = "models\hubert_base_ls960.pt"
    #ckpt_path_large =  "models\hubert_large_ll60k.pt"
    #models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path_small])
    #basemodel = models[0]
    #print(basemodel)

    transforms = BirdSetTransformsWrapper(
        task=task,
        sampling_rate = 16000,
        model_type = 'waveform',
    )

    start_t = time.time()
    """dmxcm = BirdSetDataModule(
        dataset= DatasetConfig(
            data_dir='B:\DLL\Datasets',
            dataset_name= "XCM",
            hf_path='DBD-research-group/BirdSet',
            hf_name="XCM",
            n_workers=8,
            val_split=0.2,
            task="multiclass",
            classlimit=500,
            eventlimit=1,
            sampling_rate=16000,
        ),
        transforms=transforms
    )"""

    dm = BirdSetDataModule(
        dataset= DatasetConfig(
            data_dir='B:\DLL\Datasets',
            dataset_name=dataset_name,
            hf_path='DBD-research-group/BirdSet',
            hf_name=dataset_name,
            n_workers=8,
            val_split=0.2,
            task=task,
            classlimit=500,
            eventlimit=1,
            sampling_rate=16000,
        ),
        transforms=transforms
    )

    dm.prepare_data()
    dm.setup(stage="fit")
    dm.setup(stage="test")
    train_loader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()
    test_dataloader = dm.test_dataloader()
    print(type(train_loader.dataset))
    print(train_loader.dataset)
    print(type(train_loader))
    print(type(test_dataloader.dataset))
    print(test_dataloader.dataset)
    print(type(test_dataloader))
    print(f"datamodule took {time.time() - start_t} seconds")

    #loss_func = torch.nn.BCELoss()

    num_classes = dm.num_classes

    model = Modelwrapper(basemodel, num_classes, task)
    print(model)

    #optimizer = torch.optim.Adam(model.parameters(), lr=0)

    epochs = 50

    logger = TensorBoardLogger("lightning_logs", name=f"{model_name}-{dataset_name}-{task}")#+time.strftime("%d-%m-%Y_%H-%M-%S")

    trainer = lt.Trainer(max_epochs=epochs, accelerator="gpu", devices=[1], logger=logger)
    trainer.fit(model, train_loader, val_dataloader)
    model.save_model(f"D:\\Documents\\Uni\\DLL\\Code\\models\\")

    #for epoch in range(epochs):
    #    losses = []
    #    for i, batch in enumerate(tqdm(train_loader)):
            
    #    print(f"epoch {epoch}: trainloss {torch.mean(losses)}")