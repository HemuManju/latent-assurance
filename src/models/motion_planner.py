import pytorch_lightning as pl

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class MotionPlanner(pl.LightningModule):
    def __init__(self, hparams, net, data_loader):
        super(MotionPlanner, self).__init__()
        self.h_params = hparams
        self.net = net
        self.data_loader = data_loader

        # Save hyperparameters
        self.save_hyperparameters(self.h_params)

    def forward(self, x):
        output = self.net.forward(x)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Predict and calculate loss
        self.forward(x)

        # TODO: Implement criterion
        loss = 0  # criterion(output, y)

        self.log('losses/train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Predict and calculate loss
        self.forward(x)

        # TODO: Implement criterion
        loss = 0  # criterion(output, y)

        self.log('losses/val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def train_dataloader(self):
        return self.data_loader['train_dataloader']

    def val_dataloader(self):
        return self.data_loader['val_dataloader']

    def test_dataloader(self):
        return self.data_loader['test_dataloader']

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.h_params['LEARNING_RATE'])
        ReduceLROnPlateau(
            optimizer, patience=10, factor=0.9, verbose=True
        )

        return [optimizer]
