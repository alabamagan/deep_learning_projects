r"""
Generate 3-fold cross-validation data split
"""

import numpy as np
from pytorch_med_imaging.utils.batchgenerator import GenerateTestBatch
from imblearn import under_sampling
from pathlib import Path
import pandas as pd
import re
import configparser
import pprint
from sklearn.model_selection import StratifiedKFold, train_test_split

def update_data():
    r"""Check image data directory to identify the available images."""
    id_globber = '\w{0,5}\d+'
    target_dir = Path('../../NPC_Segmentation/60.Large-Study/v1-All-Data/Normalized_2/T2WFS_TRA/01.NyulNormalized')
    data_file = Path('./Datasheet_v2.xlsx')

    # Get IDs from datasheet
    df = pd.read_excel(str(data_file), index_col=0)
    dfid = set(df.index)

    # Glob ID from target dir
    ids = []
    for f in target_dir.glob("*nii.gz"):
        mo = re.search(id_globber, f.name)
        if mo is None:
            print(f"Can't glob ID from: {f.name}")
            continue
        else:
            _id = mo.group()
            ids.append(_id)
    ids = set(ids)

    # Find differences
    miss_in_table = ids - dfid
    miss_in_folder = dfid - ids

    print(f"miss_in_table: {miss_in_table}")
    print(f"miss_in_folder: {miss_in_folder}")

def generate_batch():
    r"""Generate a 3-fold split with a distribution of NPC patients close to the distribution found in an NPC screening
    study using Epstein-Barr Virus (EBV) DNA blood test [1].

    Typically, the distribution of stage I to IV patients were around 1:2:3:2, respectively, but in that study, the NPC
     confirmed patients with screen +ve results were 8:4:4:1 respectively.

    .. note::
        1. Chan KCA, Woo JKS, King A, Zee BCY, Lam WKJ, Chan SL, Chu SWI, Mak C, Tse IOL, Leung SYM, Chan G, Hui EP, Ma
           BBY, Chiu RWK, Leung SF, van Hasselt AC, Chan ATC, Lo YMD. Analysis of Plasma Epstein-Barr Virus DNA to
           Screen for Nasopharyngeal Cancer. N Engl J Med. 2017;377(6):513-22. doi: 10.1056/NEJMoa1701717.
    """
    # Load data
    data_file = Path("./Datasheet_v2.xlsx")
    random_seed = 8929304
    df = pd.read_excel(str(data_file), index_col=0)

    # extract NPC cases only
    tstage = df['Tstage']
    nstage = df['Nstage']
    npc_patients = df.index[df['Tstage'] > 0]
    non_npc_patients = df.index[df['Tstage'] == 0]
    tstage_counts = df.loc[npc_patients]['Tstage'].value_counts().sort_index()
    print(tstage_counts)
    tstage_counts = tstage_counts.to_numpy()

    # Under sample these npc patients
    target_number = 60
    prevalance = 0.2    # Patients with +ve EBV have 10-20% of chance of having NPC
    npc_number = prevalance * target_number
    screening_distribution = pd.Series({
        1: 8 / 17.,
        2: 4 / 17.,
        3: 4 / 17.,
        4: 1 / 17.
    })
    # screening_distribution = np.ceil(screening_distribution * npc_number).astype('int')

    # For training, generate three-fold data
    tstage_ratio = screening_distribution.to_numpy()
    tstage_w_mincount = np.argmin(tstage_counts / tstage_ratio) + 1 # Find which Tstage is the bottleneck
    tstage_base_unit = tstage_counts[tstage_w_mincount - 1] / screening_distribution[tstage_w_mincount]
    dist_for_training = np.ceil(tstage_base_unit * screening_distribution).astype('int')
    print(f"dist_for_training: {dist_for_training}")

    # For validation, generate data according to screening distribution
    dist_for_validation = np.ceil(screening_distribution * target_number * prevalance).astype('int')
    print(f"dist_for_validation: {dist_for_validation}")

    rus = under_sampling.RandomUnderSampler(
        sampling_strategy = dist_for_training.to_dict(),
        random_state=random_seed
    )
    X, y = rus.fit_resample(np.arange(len(npc_patients)).reshape(-1, 1),
                            df.loc[npc_patients]['Tstage'].to_numpy().astype('int'))


    # NPC + non-NPC patinets
    new_index = non_npc_patients.tolist() + npc_patients[X].flatten().tolist()
    stats = df.loc[new_index].copy().sort_index()

    # 10% as validation data
    train, validation, _, __ = train_test_split(stats.index, stats['Tstage'], test_size=0.2, random_state=random_seed)

    # Create three fold for training data, put the rest as validation data
    splitter = StratifiedKFold(n_splits=3, random_state=random_seed, shuffle=True)
    for i, (train, test) in enumerate(splitter.split(stats.index, stats['Tstage'])):
        train, test = stats.index[train], stats.index[test]
        train_str = ','.join(train)
        test_str=','.join(test)
        pprint.pprint(train_str, width=120)
        pprint.pprint(test_str, width=120)
        CFG = configparser.ConfigParser()
        CFG.add_section('FileList')
        CFG['FileList']['training'] = train_str
        CFG['FileList']['testing'] = test_str
        CFG.write(open(f'./B{i:02d}.ini', 'w'))

    # not selected goes into validation
    not_selected = validation.tolist()

    # undersample patients with Tstage 3 and 4
    not_selected.sort()
    open('./Validation.txt', 'w').writelines('\n'.join(not_selected))

    # print summary
    pprint.pprint({
        'kfold': df.loc[train]['Tstage'].value_counts(),
        'validation': df.loc[validation]['Tstage'].value_counts()
    })


if __name__ == '__main__':
    generate_batch()
    # update_data()

