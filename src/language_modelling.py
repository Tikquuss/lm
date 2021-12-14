import torch.optim as optim
import pytorch_lightning as pl

OPTIMIZER_DIC = {"Adam": optim.Adam}

class LMLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        learning_rate: float,
        lr_factor: float,
        lr_patience: int,
        optimizer_name: str,
        label_column : str = ""
    ):
        super(LMLightningModule, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.optimizer_name = optimizer_name
        self.label_column = label_column
        # TODO
        self.decoder_start_token_id = None

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

    def _compute_loss(self, features):
        # TODO : clf_label, acc, ...
        clf_label = features.pop(self.label_column, None)
        output = self.model(**features)
        loss, logits = output.loss, output.logits
        return loss, {}
    
    def training_step(self, batch, batch_idx):
        loss, output = self._compute_loss(batch)
        output["loss"] = loss
        self.log_dict({"loss" : loss}, prog_bar=True)
        return output

    def validation_step(self, batch, batch_idx):
        loss, output = self._compute_loss(batch)
        output["val_loss"] = loss
        self.log_dict(output, prog_bar=True)
        return output

    def test_step(self, batch, batch_idx):
        loss, output = self._compute_loss(batch)
        output["test_loss"] = loss
        self.log_dict(output, prog_bar=True)
        return output

    def generate(self, input_ids):
        return self.model.generate(
            input_ids, decoder_start_token_id=self.decoder_start_token_id
        )