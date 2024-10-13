if(__name__ == "__main__"):
    from birdset.datamodule import DatasetConfig
    from birdset.datamodule.birdset_datamodule import BirdSetDataModule

    dm = BirdSetDataModule(
        dataset= DatasetConfig(
            data_dir='B:\DLL\Datasets',
            dataset_name='HSN',
            hf_path='DBD-research-group/BirdSet',
            hf_name='HSN',
            n_workers=8,
            val_split=0.2,
            task="multilabel",
            classlimit=500,
            eventlimit=5,
            sampling_rate=16000,
        ),
    )

    dm.prepare_data()