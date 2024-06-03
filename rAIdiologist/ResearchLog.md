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

| Hyperparam    | Value        |
|---------------|--------------|
| Learning Rate | 1E-7 to 5E-7 |
| Init Momentum | 0.99         |
