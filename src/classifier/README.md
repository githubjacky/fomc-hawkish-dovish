To run a expermient of a text classifier we need to:
## Setup `DataModule`

In `DataModule` we define what data we're using, how we split the training dataset into training
data and validation. Moreover, how batch of data is collated during the forward pass is also
included.


---
## Define the Class of `model`

A `model` class is the subclass of `L.LightningModule` in which we define how to stack the neural network defined in `nn.py` and what should be logged.


---
## Run `train.py`

In `train.py` we create all required objects to train a classifer, including data module, `model`
and `trainer`. Once `train.py` is running, an `optuna.Study` has been created. Note that a training
procedure encompasses three stages, i.e, training, validation, and testing.

A `trainer` object is created once the file is running and various components are activatd, e.g, `WandbLogger`, `EarlyStopping`, `ModelCheckpoint` and `TQDMProgressBar`.