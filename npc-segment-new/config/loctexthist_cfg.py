from pytorch_med_imaging.solvers import SegmentationSolver, SegmentationSolverCFG
from pytorch_med_imaging.controller import PMIController, PMIControllerCFG
from pytorch_med_imaging.networks.specialized.unet_loc_tex import UNetFCAttention_p

from npc_segment.network import UNetLocTexHistDeeper

class SegmentCFG(SegmentationSolverCFG):
    net = UNetLocTexHistDeeper(1, 2, fc_inchan=204)

    unpack_key_forward   = [('input', 'feature'), 'gt']
    unpack_key_inference = ['input' , 'feature']

    gt_keys               = ['gt']
    sigmoid_params        = {'delay' : 5         , 'stretch': 1, 'cap': 0.2}

    class_weights       = [0.01, 1.0]
    num_of_epochs       = 1
    decay_rate_LR       = 1
    decay_on_plateau    = False

    lr_sche = 'ExpoentialLR'
    lr_scheduler_dict   = {'cooldown':100, 'patience':50}

    # How to scale class weights?