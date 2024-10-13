import torch
import lightning as lt
from torchmetrics.classification.average_precision import AveragePrecision
from torchmetrics.classification.auroc import AUROC
from sklearn.metrics import average_precision_score
from torchmetrics import Accuracy, Precision
import os, time
class Modelwrapper(lt.LightningModule):
    #output from modelextractor: [32, 512, 249]
    #output from model: [32, 249, 512]
    def __init__(self, model, num_classes, task, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.task = task
        self.num_classes = num_classes
        model.eval()
        self.part = model.feature_extractor
        self.part.requires_grad_(False)
        #self.part.encoder.requires_grad_(True)
        self.lastConv = torch.nn.Conv1d(512, 1, kernel_size=1, dtype=torch.float16)
        self.activation = torch.nn.LeakyReLU()
        self.linear = torch.nn.Linear(249, num_classes, dtype=torch.float16)
        self.sigmoid = torch.nn.Sigmoid()
        self.loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([num_classes-1]*num_classes)) if task == "multilabel" else torch.nn.CrossEntropyLoss()
        self.precision_func = Precision(task, num_classes=num_classes, num_labels=num_classes)
        self.accuracy_func = Accuracy(task, num_classes=num_classes, num_labels=num_classes)
        self.auroc_func = AUROC(task=task, num_labels=num_classes, num_classes=num_classes)
        self.cmap_func = AveragePrecision(task=task, num_classes=num_classes, num_labels=num_classes, average="macro") #from https://github.com/DBD-research-group/BirdSet/blob/main/birdset/modules/metrics/multiclass.py
        self.sdevice = "cuda:1"
        self.to(self.sdevice)

    def forward(self, x):
        input = x.squeeze().half().to(self.device)
        output = self.part(input)
        convout = self.activation(self.lastConv(output).squeeze())
        linout = self.linear(convout)
        return linout
    
    def getFeatures(self, batch):
        input = batch["input_values"].squeeze().half().to(self.device)
        output = self.part(input)
        return output
    
    def get_proba_sigmoid(self, batch):
        pred = self(batch["input_values"])
        proba = self.sigmoid(pred)
        return proba

    def get_proba_softmax(self, batch):
        pred = self(batch["input_values"])
        proba = pred.softmax(dim=1)
        return proba
        
    def configure_optimizers(self):
        #return torch.optim.Adam([self.lastConv.parameters() + self.linear.parameters()], lr=1e-3)
        #return torch.optim.SGD(self.parameters())
        optimizer = torch.optim.Adam(self.parameters(), eps=1e-6, lr=0.001)
        return optimizer
        scheduler = CosineAnnealingLR(optimizer, T_max=281, eta_min=0.0000001)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler" : scheduler, "interval": "step", "frequency": 25}}

    def training_step(self, batch):
        pred = self.get_proba_sigmoid(batch) if self.task == "multilabel" else self.get_proba_softmax(batch)
        labels = batch["labels"]
        #pred = torch.rand(size=(32,num_classes), dtype=torch.float16).to("cuda:1")
        #print("batch", batch["labels"])
        if(self.task=="multiclass"):
            labels = torch.nn.functional.one_hot(batch["labels"], self.num_classes).to(torch.float16)
        #print("onehot: ", one_hot_label)
        loss = self.loss_func(pred, labels)
        #logging possible
        self.log("training_loss", loss, )
        #print(loss)
        if(torch.isnan(loss)):
            print(pred)
            print(batch["labels"].shape)
            print(pred.shape)
            print(f"loss got nan")
            exit(-1)
        #losses.append(loss)
        return loss
    
    def validation_step(self, batch):
        self.eval()
        pred = self.get_proba_sigmoid(batch) if self.task == "multilabel" else self.get_proba_softmax(batch)
        labels = batch["labels"]
        self.log("test_accuracy",  self.accuracy_func(pred, labels))
        self.log("test_auc", self.auroc_func(pred, labels.to(torch.int)))
        if(self.task=="multilabel"):
            self.log("test_cMAP", average_precision_score(labels.cpu(), pred.cpu(), average="macro"))
        else:
            self.log("test_cMAP", self.cmap_func(pred, labels.to(torch.int)))
        self.log("test_precision", self.precision_func(pred,labels.to(torch.int)))
        self.train()

    def test_step(self, batch):
        pass

    def save_model(self, path, additional_content=None):
        if(additional_content is None):
            additional_content = {}
        if(path != "" and not os.path.isdir(path)):
            os.mkdir(path)
        path = path+self.__class__.__name__+time.strftime("%d-%m-%Y-%H-%M-%S", time.localtime())+".pt"
        additional_content["num_classes"] = self.num_classes
        additional_content["task"] = self.task
        additional_content["state_dict"] = self.state_dict()
        torch.save(additional_content, path)
        return path
    
    def load_model(path, model):
        load = torch.load(path)
        #print(load)
        try:
            model = Modelwrapper(model=model, num_classes=load["num_classes"], task=load["task"])
            #model = Modelwrapper(model=model, num_classes=load["num_classes"], task="multilabel")
            model.load_state_dict(load["state_dict"])
        except KeyError:
            print("File does not contain the right keys")
            return None
        return model