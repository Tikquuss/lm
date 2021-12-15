import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
import math
from loguru import logger
from .metrics import get_compute_metrics_lm, get_compute_metrics_clf

OPTIMIZER_DIC = {"Adam": optim.Adam}

class PredLayer4Classification(nn.Module):
    """Prediction layer"""
    def __init__(self, d_model, n_labels, n_layers = 1, intermediate_dim = None, 
                criterion = None, dropout = 0, multi_label = False):
        super().__init__()
        assert n_labels >= 2
        assert n_layers > 0 and (n_layers == 1 or intermediate_dim is not None) 
        self.n_labels = 1 if n_labels == 2 and not multi_label else n_labels
        net = [nn.Dropout(dropout) if dropout != 0 else nn.Identity()]
        if n_layers == 1 :
            net.append(nn.Linear(d_model, self.n_labels))
        else :
            net.extend([
                nn.Linear(d_model, intermediate_dim), 
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            for _ in range(n_layers-2) :
                net.extend([
                    nn.Linear(intermediate_dim, intermediate_dim), 
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
            net.append(nn.Linear(intermediate_dim, self.n_labels))
        
        self.proj = nn.Sequential(*net)
                
        if criterion is not None :
            self.criterion = criterion
        else :
            if self.n_labels == 1 or multi_label :
                #self.criterion = nn.BCEWithLogitsLoss().to(device)
                self.criterion = F.binary_cross_entropy_with_logits
            else :
                #self.criterion = nn.CrossEntropyLoss(weight=weight, reduction='mean').to(device)
                self.criterion = F.cross_entropy
            
    def forward(self, x, y, weights = None, weight_out = None):
        """
        Compute the loss and the scores
        x : (bs, d_model)
        y : 
            - torch.FloatTensor(bs, 1) if binary classification (n_labels = 2 and not multi-label classification)
            - torch.FloatTensor(bs, n_labels) if multi-label classification
            - torch.LongTensor(bs,) if binary multi-class classification (n_labels > 2 and not multi-label classification)
        """
        #x = F.normalize(input = x, p=2, dim=1, eps=1e-12, out=None)
        x = self.proj(x)
        #x = F.dropout(x, p=0.1, training=self.training)
        scores = x.view(-1, self.n_labels)
        if weight_out is None :
            loss = self.criterion(scores, y, weight=weights)
        else :
            loss = weight_out*self.criterion(scores, y, weight=weights, reduction="none")
            loss = loss.mean()

        return scores, loss

    def get_scores(self, x):  
        return self.proj(x)
    
class LMLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        task : str,
        learning_rate: float,
        lr_factor: float,
        lr_patience: int,
        optimizer_name: str,
        decoder_start_token_id : int = None,
        clf_params : dict = {},
        
    ):
        super(LMLightningModule, self).__init__()
        assert not clf_params or all([k in clf_params for k in ["label_column", "n_labels"]])
        self.model = model
        self.learning_rate = learning_rate
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.optimizer_name = optimizer_name
        self.label_column = clf_params.pop("label_column", None)
        self.clf_params = clf_params
        self.decoder_start_token_id = decoder_start_token_id
        self.compute_metrics_lm = get_compute_metrics_lm(task)
        self.build_clf_compute_loss()
        self.is_clf_token = task == "mlm"
        self.build_hidden_for_clf()
        self.lambda_clf = 1.0

    def configure_optimizers(self):
        optimizer = OPTIMIZER_DIC[self.optimizer_name](
            self.model.parameters(), lr=self.learning_rate
        )
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=self.lr_factor, patience=self.lr_patience
        )
        output = {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }
        return output
    
    def comupte_perplexity(self, loss):
        try:
            return math.exp(loss)
        except OverflowError:
            return float("inf")
            
    def _compute_loss(self, features, prefix=""):
        prefix = "%s_"%prefix if prefix else prefix
        clf_label = features.pop(self.label_column, None)
        mask_token_index = features.pop("mask_token_index", None)
        output = self.model(**features)
        loss, logits = output.loss, output.logits
        output = self.compute_metrics_lm(features["labels"], logits, mask_token_index, prefix)
        output["%sloss"%prefix] = loss.item()
        output["%sppl"%prefix] = self.comupte_perplexity(loss)
        hidden = self.get_hidden_for_clf(features)
        clf_output = self.clf_compute_loss(clf_label, hidden, prefix)
        clf_loss = clf_output.pop("loss", torch.tensor(0.))
        output = {**output, **clf_output}
        return loss + self.lambda_clf*clf_loss, output
    
    def training_step(self, batch, batch_idx):
        loss, output = self._compute_loss(batch, prefix="train")
        self.log_dict(output, prog_bar=True)
        output["loss"] = loss
        return output

    def validation_step(self, batch, batch_idx):
        loss, output = self._compute_loss(batch, prefix="val")
        #output["val_loss"] = loss
        self.log_dict(output, prog_bar=True)
        return output

    def test_step(self, batch, batch_idx):
        loss, output = self._compute_loss(batch, prefix="test")
        #output["test_loss"] = loss
        self.log_dict(output)
        return output

    def generate(self, input_ids):
        return self.model.generate(
            input_ids, decoder_start_token_id=self.decoder_start_token_id
        )
        
    def build_hidden_for_clf(self):
        if not self.clf_params :
            def get_hidden_for_clf(features) :
                return None
        elif self.is_clf_token :
            def get_hidden_for_clf(features) :
                x = self.model.bert(
                    input_ids = features["input_ids"],
                    attention_mask = features["attention_mask"],
                    token_type_ids= features.get("token_type_ids", None))  
                latent = x['last_hidden_state']# (bs, x_len, d_model)
                latent = latent[:, 0] # (bs, d_model) ~ [CLS]
                return latent
        else :
            def get_hidden_for_clf(features) :
                x = self.model.bert(
                    input_ids = features["input_ids"],
                    attention_mask = features["attention_mask"],
                    token_type_ids= features.get("token_type_ids", None))  
                
                latent = x['last_hidden_state']# (bs, x_len, d_model)
                latent = torch.sigmoid(latent)
                latent = torch.sum(latent, dim=1)  # (bs, d_model)
                return latent
            
        self.get_hidden_for_clf = get_hidden_for_clf
    
    def build_clf_label_coverter(self, n_labels : int, multi_label : bool = False):
        assert n_labels >= 2
        if n_labels == 2 and not multi_label :
            def converter(y):
                #return y.float().unsqueeze(1)
                return torch.FloatTensor(y).unsqueeze(1)
        elif not multi_label :
            def converter(y):
                #return y.long()
                return torch.LongTensor(y)
        else :
            def converter(y):
                #return y.float()
                return torch.FloatTensor(y)
            
        self.clf_label_coverter = converter
    
    def build_clf_compute_loss(self):
        if self.clf_params :
            logger.info("PredLayer4Classification building ...")
            clf_params = self.clf_params
            n_labels = clf_params["n_labels"]
            multi_label = clf_params.get("multi_label", False)
            clf_params["d_model"] = clf_params.get("d_model", 
                                self.model.config.to_dict().get('hidden_size', self.model.lm_head.in_features)
                                            )
            self.clf = PredLayer4Classification(**clf_params)
            
            self.build_clf_label_coverter(n_labels, multi_label)
            compute_metrics_clf = get_compute_metrics_clf(n_labels)

            def clf_compute_loss(clf_label, hidden, prefix=""):
                scores, loss = self.clf(x = hidden, 
                                        y = self.clf_label_coverter(clf_label).to(hidden.device)
                                        )
                output = {}
                output = compute_metrics_clf(torch.LongTensor(clf_label), scores, prefix)
                output["loss"] = loss
                output['%s%s'%(prefix,"clf_loss")] = loss.item()
                return output
        else :
            def clf_compute_loss(clf_label, hidden, prefix=""):
                return {}
            
        self.clf_compute_loss = clf_compute_loss