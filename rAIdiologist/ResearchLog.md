# Project target

This project aims to discriminate patients with and without NPC, with some of the subjects without NPC but some degree of benign hyperplasia.

# Network

A network with a CNN encoder and a RNN decoder is proposed. The CNN encode the images slice-by-slice and provide a weighting for each slices as attention, the RNN decoder interprets the encoded deep features and consider them as a sequence of features, with `len(seq) = num_slice - 2` by excluding the top and bottom slice. These slices are excluded because the input kernel of the CNN is 3 by 3 by 3, meaning the top and bottom slices are padded.

# Training Recipies

## CNN

Currently, it seems that the old SWRAN performed the best, but its implementation is not accurately representing that of the original paper. The SWRAN was re-implemented to better match the original framework, but the results were not as good for some reason. 

### Hyper-parameters

#### Using ADAM

| Hyperparam       | Value        |
|------------------|--------------|
| Learning Rate    | 1E-5 to 5E-5 |
| Initial momentum | 0.99         |
| Num Epochs       | 120          |

#### Using OnecycleLR with SGD

| Hyperparam       | Value |
|------------------|-------|
| Learning Rate    | 1E-5  |
| `lr_sche_args`   | 0.03  |
| `cycle_momentum` | True  |
| `three_phase`    | True  | 
| `div_factor`     | 100   |
| Num Epochs       | 50    |

## RNN

The possibiltiy of using transformer is being tested currently. The input of the transformer would be the CNN encoded deep features with some dropouts. The transformer is first trained with the pre-trained CNN frozen (i.e., training mode 1). The hyper-parameters are as follow

#### Using ADAM

| Hyperparam    | Value                 |
|---------------|-----------------------|
| Learning Rate | ~~1E-7 to 5E-7~~ 1E-5 |
| Init Momentum | 0.99                  |

**[04/06/2024-Matthew]**:  Note that I found I previously set the learning rate of cnn and rnn to be weighted using the code below, it is therefore estimated the rate between RNN and CNN is 2:10. I now implemented a more general option to control this by allowing an attribute `lr_weights` in the network that modifies the behavior of `optimizer_set_params()` method.  

```python
# Method of rAIdiologistSolver
def optimizer_set_params(self):
    r"""The learning rate of RNN and CNN are inherently different, and we can't use the same for both and train
    them together. Therefore, this method is overrided to tune the factor."""
    if not isinstance(self.get_net(), rAIdiologist):
        super().optimizer_set_params()
    else:
        net = self.net
        lstm_factor = os.getenv('lstm_initlr_factor', 5)
        assert isinstance(net, rAIdiologist), "Incorrect network setting."
        args = [
            {"params": net.cnn.parameters(), "lr": self.init_lr},
            {"params": net.lstm_rater.parameters(), "lr": self.init_lr * 5},
        ]
    ...

```

**[13/06/2024-Matthew]**: The network is now given a new attribute "lr_weights", which can be used to control the weight
factor of each individual module. Implementation of changes was mainly in solver, when setting the network into the 
optimizer. If both CNN and RNN are being trained from scratch, the convergence is difficult. If only RNN is being 
trained, you can use the parameters above. Otherwise, there's high risk of overfitting.

**[07/07/2024-Matthew]**: Although the network converges with the new configuration, the performance still tops at around
0.966 AUC, this is less than the previous versions of the network. Therefore, I am adding back the confidence factor 
that will help merge the CNN and RNN outputs.