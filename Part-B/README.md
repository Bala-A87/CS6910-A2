# CS6910 A2 Part B

Fine-tuning a pre-trained convolutional neural network.

## Directory contents

- [learn.py](./learn.py): Contains helper functions `train` and `predict`
- [metrics.py](./metrics.py): PyTorch Module implementations of metrics for model evaluation. Only `CategoricalAccuracy` has been implemented as it is the only one used for this assignment
- **[train.py](./train.py)**: Training script to fine-tune a *ResNet50* model with the output layer modified to match the *iNaturalist* dataset, optionally logging the run on WandB. More details on the script follow in the next section
- [part-b.ipynb](./part-b.ipynb): Jupyter notebook used for testing out implementation details before conversion to scripts, mostly just random trials

## Training script

The training script, [train.py](./train.py) provides an abstraction for loading data, loading the model with default weights, freezing the layers and replacing the output layer with one of the correct width (10), training it and evaluating it on the train (and optionally) test data, optionally logging the training on WandB. Multiple command line arguments are supported, described as follows:
| Argument flag | Description | Default |
|:-:|:-:|:-:|
| `-da`, `--data_aug` | Adds augmented data to train and val datasets if passed | N/A |
| `-bs`, `--batch_size` | Batch size used for training and evaluation | 128 |
| `dev`, `--device` | Device on which compute is done | cuda |
| `-lr`, `--learning_rate` | Learning rate used for training | 1e-4 |
| `-wd`, `--weight_decay` | Weight decay (L2 reg) used during training | 1e-3 |
| `-e`, `--epochs` | Number of epochs to train for | 5 |
| `-we`, `--wandb_entity` | WandB user where to track runs. Asks for login if not passed and project is passed | None |
| `-wp`, `--wandb_project` | WandB project where to track runs. Not tracked if not passed | None |
| `-run`, `--run_test` | Evaluates trained model on test data if passed | N/A |
