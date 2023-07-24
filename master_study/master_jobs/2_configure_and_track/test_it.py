# %%
import xtrack as xt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import configure_and_track as configure_and_track

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
collider.survey_b1 = survey_b1
collider.survey_b2 = survey_b2

# %% filling scheme computation
# import a yaml file in a dictionary
import yaml
with open('config.yaml') as file:
    config = yaml.safe_load(file)
config_sim = config['config_simulation']
config_collider = config['config_collider']

with open('../1_build_distr_and_collider/config.yaml') as file:
    config_generation_1 = yaml.safe_load(file)
config_mad = config_generation_1['config_mad']

collider.config = config_collider
collider.config_mad = config_mad

# %%

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
collider.bb_schedule = bb_schedule
# %%
bb_schedule_b1 = bb_schedule.b1.bb_schedule.loc[b1_bunch_to_track]
bb_schedule_b2 = bb_schedule.b2.bb_schedule.loc[b2_bunch_to_track]

print('\nBunch to track in Beam 1:')
print(bb_schedule_b1)
print('\nBunch to track in Beam 2:')
print(bb_schedule_b2)


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
# %% To check if there are disconnected knobs in the configuration
for ii in config_collider['config_knobs_and_tuning']['knob_settings'].keys():
    if len(collider.vars[ii]._find_dependant_targets())==1:
        print(ii)
# %% Some plots
# collider.vars['on_alice_normalized'] = 1
# collider.vars['on_lhcb_normalized'] = -1
# collider.vars['beambeam_scale'] = 1
# collider.vars['on_x8h'] = -170
# collider.vars['on_sep2h'] = 100
import numpy as np
from scipy import constants

def compute_separation(collider, ip='ip1', beam_weak = 'b1', verbose=True):

    twiss_b1 = collider['lhcb1'].twiss(matrix_stability_tol=1.01)
    twiss_b2 = collider['lhcb2'].twiss(matrix_stability_tol=1.01).reverse()

    if beam_weak == 'b1':
        beam_strong = 'b2'
        twiss_weak = twiss_b1
        twiss_strong = twiss_b2
        survey_weak = collider.survey_b1
        survey_strong = collider.survey_b2
    else:
        beam_strong = 'b1'
        twiss_weak = twiss_b2
        twiss_strong = twiss_b1
        survey_weak = collider.survey_b2
        survey_strong = collider.survey_b1

    assert (collider.config_mad['beam_config']['lhcb1']['beam_energy_tot'] 
            ==
            collider.config_mad['beam_config']['lhcb1']['beam_energy_tot'])

    energy = collider.config_mad['beam_config']['lhcb1']['beam_energy_tot']
    # gamma relativistic of a proton at 7 TeV
    gamma_rel = energy/(constants.physical_constants['proton mass energy equivalent in MeV'][0]/1000)
    # beta relativistic of a proton at 7 TeV
    beta_rel = np.sqrt(1-1/gamma_rel**2)

    emittance_strong_nx = collider.config['config_beambeam']['nemitt_x']
    emittance_strong_ny = collider.config['config_beambeam']['nemitt_y']

    emittance_weak_nx = collider.config['config_beambeam']['nemitt_x']
    emittance_weak_ny = collider.config['config_beambeam']['nemitt_y']

    emittance_strong_x = emittance_strong_nx/gamma_rel/beta_rel
    emittance_strong_y = emittance_strong_ny/gamma_rel/beta_rel

    emittance_weak_x = emittance_weak_nx/gamma_rel/beta_rel
    emittance_weak_y = emittance_weak_ny/gamma_rel/beta_rel

    #ax = coordinates['x_sig']
    #ay = coordinates['y_sig']
    survey_filtered = {}
    twiss_filtered = {}
    my_filter_string = f'bb_(ho|lr)\.(r|l|c){ip[2]}.*'
    survey_filtered[beam_strong] = survey_strong[f'ip{ip[2]}'][['X','Y','Z'], my_filter_string]
    survey_filtered[beam_weak] = survey_weak[f'ip{ip[2]}'][['X','Y','Z'], my_filter_string]
    twiss_filtered[beam_strong] = twiss_strong[:, my_filter_string]
    twiss_filtered[beam_weak] = twiss_weak[:, my_filter_string]

    s = survey_filtered[beam_strong]['Z']
    d_x_weak_strong_in_meter = (
        twiss_filtered[beam_weak]['x'] - twiss_filtered[beam_strong]['x'] +
        survey_filtered[beam_weak]['X']- survey_filtered[beam_strong]['X']
        )
    d_y_weak_strong_in_meter = (
        twiss_filtered[beam_weak]['y'] - twiss_filtered[beam_strong]['y'] +
        survey_filtered[beam_weak]['Y']- survey_filtered[beam_strong]['Y']
        )

    sigma_x_strong = np.sqrt(twiss_filtered[beam_strong]['betx']*emittance_strong_x)
    sigma_y_strong = np.sqrt(twiss_filtered[beam_strong]['bety']*emittance_strong_y)

    sigma_x_weak = np.sqrt(twiss_filtered[beam_weak]['betx']*emittance_weak_x)
    sigma_y_weak = np.sqrt(twiss_filtered[beam_weak]['bety']*emittance_weak_y)

    dx_sig = d_x_weak_strong_in_meter/sigma_x_strong
    dy_sig = d_y_weak_strong_in_meter/sigma_y_strong

    A_w_s = sigma_x_weak/sigma_y_strong
    B_w_s = sigma_y_weak/sigma_x_strong

    fw = 1 
    r = sigma_y_strong/sigma_x_strong

    my_dict = { 'twiss_filtered':twiss_filtered,
                'survey_filtered':survey_filtered,
                's':s,
                'dx_sig':dx_sig,
                'dy_sig':dy_sig,
                'A_w_s':A_w_s,
                'B_w_s':B_w_s,
                'fw':fw,
                'r':r,
                'emittance_strong_x':emittance_strong_x,
                'emittance_strong_y':emittance_strong_y,
                'emittance_weak_x':emittance_weak_x,
                'emittance_weak_y':emittance_weak_y,
                'gamma_rel':gamma_rel,
                'beta_rel':beta_rel,
                'energy':energy,
                'my_filter_string':my_filter_string,
                'beam_weak':beam_weak,
                'beam_strong':beam_strong,
                'ip':ip,
    }
    if verbose:
        print(my_dict)
    return my_dict
my_dict = {}
for my_ip in ['ip1','ip2','ip5','ip8']:
    my_dict[my_ip] = compute_separation(collider, ip=my_ip)
# %%

def plot_orbits(ip_dict):
    plt.figure()
    plt.title(f'IP{ip_dict["ip"][2]}')
    beam_weak = ip_dict['beam_weak']
    beam_strong = ip_dict['beam_strong']
    twiss_filtered = ip_dict['twiss_filtered']
    plt.plot(ip_dict['s'], twiss_filtered[beam_weak]['x'], 'ob', label=f'x {beam_weak}')
    plt.plot(ip_dict['s'],  twiss_filtered[beam_strong]['x'], 'sb', label=f'x {beam_strong}')
    plt.plot(ip_dict['s'], twiss_filtered[beam_weak]['y'], 'or', label=f'y {beam_weak}')
    plt.plot(ip_dict['s'],  twiss_filtered[beam_strong]['y'], 'sr', label=f'y {beam_strong}')
    plt.xlabel('s [m]')
    plt.ylabel('x,y [m]')
    plt.legend()
    plt.grid(True)


def plot_separation(ip_dict):
    plt.figure()
    plt.title(f'IP{ip_dict["ip"][2]}')
    plt.plot(ip_dict['s'], np.abs(ip_dict['dx_sig']), 'ob', label='x')
    plt.plot(ip_dict['s'], np.abs(ip_dict['dy_sig']), 'sr', label='y')
    plt.xlabel('s [m]')
    plt.ylabel('separation in x,y [$\sigma$]')
    plt.legend()
    plt.grid(True)

for my_ip in ['ip1','ip2','ip5','ip8']:
    plot_orbits(my_dict[my_ip])
    plot_separation(my_dict[my_ip])
# %% Compute the luminosity

collider.vars['on_alice_normalized'] = 1
collider.vars['on_lhcb_normalized'] = -1
collider.vars['beambeam_scale'] = 1
collider.vars['on_x8h'] = -170
collider.vars['on_sep2h'] = .1401

from xtrack import lumi
def compute_luminosity(collider, ip='ip1', sigma_tot = 81e-27, crab=False, matrix_stability_tol=1.01, verbose=True):
    '''
    Compute the LHC luminosity in cm^-2 s^-1 and the multiplicity in a given IP

    Parameters
    ----------
    collider : xtrack.multiline
        The collider object. It assumens the it has the config and bb_schedule attributes
    my_ip : str, optional
        The IP to compute the luminosity in, by default 'ip1'
    sigma_tot : float, optional
        The total cross section, by default 81e-27 cm^-2
    crab : bool, optional
        If True, the crab cavities are on, by default False
    '''
    twiss_b1 = collider['lhcb1'].twiss(matrix_stability_tol=matrix_stability_tol)
    twiss_b2 = collider['lhcb2'].twiss(matrix_stability_tol=matrix_stability_tol) 
    assert twiss_b1.T_rev0 == twiss_b1.T_rev0
    my_dict = {'ip1':collider.bb_schedule.n_coll_ATLAS,
               'ip2':collider.bb_schedule.n_coll_ALICE,
               'ip5':collider.bb_schedule.n_coll_ATLAS,
               'ip8':collider.bb_schedule.n_coll_LHCb,
            }

    colliding_bunches = my_dict[ip]
    my_luminosity = lumi.luminosity_from_twiss(
        colliding_bunches,
        collider.config['config_beambeam']['num_particles_per_bunch'],
        ip,  
        collider.config['config_beambeam']['nemitt_x'],
        collider.config['config_beambeam']['nemitt_y'],
        collider.config['config_beambeam']['sigma_z'],
        twiss_b1,
        twiss_b2,
        crab=crab,                          
    )
    my_pileup = my_luminosity*sigma_tot/colliding_bunches*twiss_b1.T_rev0 
    if verbose:
        print(f'IP{ip[2]}')
        print(f'Luminosity: {my_luminosity:.2e} cm^-2 s^-1')
        print(f'Pile-up: {my_pileup:.2e}')
        print(f'Number of colliding bunches in {ip}: {colliding_bunches}')
        print(f'Number of particles per bunch: {collider.config["config_beambeam"]["num_particles_per_bunch"]:.2e}') 
        if ip == 'ip2':
            print(f'ALICE polarity: {collider.vars["on_alice_normalized"]._value:.0f} ')
        if ip == 'ip8':
                print(f'LHCb polarity: {collider.vars["on_lhcb_normalized"]._value:.0f}')

        print('\n')
    return my_luminosity, my_pileup

my_luminosity = {}
my_pileup = {}
for my_ip in (['ip1','ip2','ip5','ip8']):
    my_luminosity[my_ip], my_pileup[my_ip] = compute_luminosity(collider, ip=my_ip)

# %%
