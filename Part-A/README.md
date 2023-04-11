# CS6910 A2 Part A

Training a convolutional neural network from scratch.

## Directory contents

- [model.py](./model.py): The implementation of the CNN architecture, `ConvNet`, in PyTorch, with helper functions `train` and `predict`
- [metrics.py](./metrics.py): PyTorch Module implementations of metrics for model evaluation. Only `CategoricalAccuracy` has been implemented as it is the only one used for this assignment
- **[train.py](./train.py)**: Training script to train a ConvNet with the given specifications on the *iNaturalist* dataset, optionally logging the run on WandB. More details on the script follow in the next section
- [sweep.py](./sweep.py): Helper script to run sweeps on WandB. Not required for the assignment and only made for convenience
- [models](./models/): Contains saved (trained) models from runs for later usage/analysis. Currently only contains a trained model of the best configuration obtained
- [part-a.ipynb](./part-a.ipynb): Jupyter notebook used for testing out implementation details before conversion to scripts, mostly just random trials

## Training script

The training script, [train.py](./train.py) uses the implementation of the model and helper functions to provide an abstract script for loading data, building the model, training it and evaluating it on the train (and optionally) test data, optionally logging the training on WandB. Multiple command line arguments are supported, described as follows:
| Argument flag | Description | Default |
|:-:|:-:|:-:|
| `-da`, `--data_aug` | Adds augmented data to train and val datasets if passed | N/A |
| `-bs`, `--batch_size` | Batch size used for training and evaluation | 128 |
| `dev`, `--device` | Device on which compute is done | cuda |
| `-f`, `--filters` | Number of conv filters in each conv layer | [128, 128, 128, 128, 128] |
| `-k`, `--kernel_size` | Kernel sizes in each conv layer | [3, 3, 3, 3, 3] |
| `-w`, `--width` | Width of hidden dense layer | 128 |
| `-do`, `--dropout` | Dropout rate to use after the hidden dense layer | None |
| `-ac`, `--activation_conv` | Activation function to use for the conv layers. One of relu, gelu, silu, mish | silu |
| `-ad`, `--activation_dense` | Activation function to use after the hidden dense layer. One of relu, gelu, silu, mish, sigmoid, tanh | relu |
| `-ps`, `--pool_size` | The maxpool kernel size for the last maxpool layer | 2 |
| `-lr`, `--learning_rate` | Learning rate used for training | 1e-4 |
| `-wd`, `--weight_decay` | Weight decay (L2 reg) used during training | 1e-3 |
| `-bn`, `--batch_norm` | Adds batch normalization after each conv-activation-pool layer bunch if passed | N/A |
| `-e`, `--epochs` | Number of epochs to train for | 10 |
| `-we`, `--wandb_entity` | WandB user where to track runs. Asks for login if not passed and project is passed | None |
| `-wp`, `--wandb_project` | WandB project where to track runs. Not tracked if not passed | None |
| `-run`, `--run_test` | Evaluates trained model on test data if passed | N/A |
| `-save`, `--save_model` | Saves trained model's state_dict in the directory `models` if passed | N/A |

The best configuration has been provided as default parameters. 