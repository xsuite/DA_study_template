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
    n_lr_per_side=25)    

for ii,zz in zip([bb_schedule.b1,bb_schedule.b2],['Beam 1','Beam 2']):
    my_bb_schedule= ii.bb_schedule.sort_values(by=['collides in ATLAS/CMS',
                                                'collides in LHCB',
                                                'collides in ALICE',
                                                '# of LR in ATLAS/CMS', 
                                                '# of LR in ALICE', 
                                                '# of LR in LHCB',
                                                ], ascending=False)   

    print(f'Suggested bunch ID for {zz}: {my_bb_schedule.index[0]}') 
# %%
bb_schedule_b1 = bb_schedule.b1.bb_schedule.loc[b1_bunch_to_track]
bb_schedule_b2 = bb_schedule.b2.bb_schedule.loc[b2_bunch_to_track]

print('\nBunch to track in Beam 1:')
print(bb_schedule_b1)
print('\nBunch to track in Beam 2:')
print(bb_schedule_b2)

# %% Compute the luminosity
from xtrack import lumi
assert twiss_b1.T_rev0 == twiss_b1.T_rev0

for ii, colliding_bunches in zip(['ip1','ip2','ip5','ip8'],
                                [bb_schedule.n_coll_ATLAS,
                                 bb_schedule.n_coll_ALICE,
                                 bb_schedule.n_coll_ATLAS,
                                 bb_schedule.n_coll_LHCb]):
    aux = lumi.luminosity_from_twiss(
        colliding_bunches,
        config_collider['config_beambeam']['num_particles_per_bunch'],
        ii,  
        config_collider['config_beambeam']['nemitt_x'],
        config_collider['config_beambeam']['nemitt_y'],
        config_collider['config_beambeam']['sigma_z'],
        twiss_b1,
        twiss_b2,
        crab=False,                          
    )

    sigma_tot = 81e-27 # cm^2
    print(f'Luminosity in {ii}: {aux:.2e} cm^-2 s^-1')
    # compute pile-up from luminosity
    print(f'Pile-up in {ii}: {aux*sigma_tot/colliding_bunches*twiss_b1.T_rev0:.2e}\n')
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
# %%
for ii in config_collider['config_knobs_and_tuning']['knob_settings'].keys():
    if len(collider.vars[ii]._find_dependant_targets())==1:
        print(ii)
# %%
