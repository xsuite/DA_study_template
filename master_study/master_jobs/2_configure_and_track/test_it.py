# %%
import xtrack as xt
import numpy as np
import pandas as pd

# add current folder to path
# import sys
# sys.path.append(os.getcwd())


import configure_and_track as configure_and_track

# %%
collider = xt.Multiline.from_json('collider.json')

# %%
collider.build_trackers()

# %%
# collider.vars['beambeam_scale'] = 0 
twiss_b1 = collider['lhcb1'].twiss()
twiss_b2 = collider['lhcb2'].twiss().reverse()

# %%
survey_b1 = {}
survey_b2 = {}

for my_ip in [1,2,5,8]:
    print(f'Survey for IP{my_ip}...')
    survey_b1[f'ip{my_ip}'] = collider['lhcb1'].survey(element0=f'ip{my_ip}')
    survey_b2[f'ip{my_ip}'] = collider['lhcb2'].survey(element0=f'ip{my_ip}').reverse()
# collider.vars['beambeam_scale'] = 1

# %% filling scheme computation
config, config_sim, config_collider = configure_and_track.read_configuration()

filling_scheme = (config_collider['config_beambeam']
                                 ['mask_with_filling_pattern']
                                 ['pattern_fname'])

b1_bunch_to_track = (config_collider['config_beambeam']
                                 ['mask_with_filling_pattern']
                                 ['i_bunch_b1'])
b2_bunch_to_track = (config_collider['config_beambeam']
                                 ['mask_with_filling_pattern']
                                 ['i_bunch_b2'])
import fillingpatterns as fp
bb_schedule = fp.FillingPattern.from_json(filling_scheme)
bb_schedule.b1.n_bunches
bb_schedule.b2.n_bunches

bb_schedule.n_coll_ATLAS
bb_schedule.n_coll_LHCb
bb_schedule.n_coll_ALICE

bb_schedule.compute_beam_beam_schedule(
    n_lr_per_side=[25, 20, 25, 20])    

bb_schedule.b1.bb_schedule.loc[b1_bunch_to_track]
bb_schedule.b2.bb_schedule.loc[b2_bunch_to_track]


# %% Compute the luminosity
from xtrack import lumi

lumi.luminosity_from_twiss(
    bb_schedule.n_coll_ATLAS,
    1.6e11,
    'ip5',  
    2.2e-6,
    2.2e-6,
    0.09,
    twiss_b1,
    twiss_b2,
    crab=False,                          
)
# %%
for my_ip in ['on_alice_normalized','on_lhcb_normalized']:
    print(f'*****************\nValues for {my_ip} (polarity):')
    print(collider.vars[my_ip]._value)
    print(f'*****************\n')

# %%
for my_ip in [1,2,5,8]:
    print(f'*****************\nValues for IP{my_ip}:')
    my_df = []
    for my_table, my_beam in zip([twiss_b1, twiss_b2],[1,2]):
        my_df.append(my_table[
            ['x', 'y', 'px', 'py', 'betx', 'bety', 'alfx', 'alfy'],
            f'ip{my_ip}'].to_pandas())
        my_df[-1].index = [f'B{my_beam}']
    print(pd.concat(my_df, axis=0).transpose())
    print(f'*****************\n')

# %%
for ii in collider.vars.get_independent_vars():
    if 'beam' in ii:
        print(ii)
# %%
collider.vars['beambeam_scale']._value 