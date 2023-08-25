# %%
import xtrack as xt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import configure_and_track as configure_and_track

# %%
#collider = xt.Multiline.from_json('../collider.json')
collider = xt.Multiline.from_json('/home/sterbini/2023_07_17_pp_reference_run/example_DA_study/master_study/scans/test/base_collider/xtrack_0000/collider.json')


collider['lhcb1'].twiss_default['matrix_stability_tol'] = 100
collider['lhcb2'].twiss_default['matrix_stability_tol'] = 100

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
with open('/home/sterbini/2023_07_17_pp_reference_run/example_DA_study/master_study/scans/test/base_collider/xtrack_0000/config.yaml') as file:
    config = yaml.safe_load(file)
config_sim = config['config_simulation']
config_collider = config['config_collider']

with open('/home/sterbini/2023_07_17_pp_reference_run/example_DA_study/master_study/scans/test/base_collider/config.yaml') as file:
    config_generation_1 = yaml.safe_load(file)
config_mad = config_generation_1['config_mad']

collider.config = config_collider
collider.config_mad = config_mad

# %%

filling_scheme = (collider.config['config_beambeam']
                                 ['mask_with_filling_pattern']
                                 ['pattern_fname'])
b1_bunch_to_track = (collider.config['config_beambeam']
                                 ['mask_with_filling_pattern']
                                 ['i_bunch_b1'])
b2_bunch_to_track = (collider.config['config_beambeam']
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
collider.vars['beambeam_scale'] = 1
twiss_b1 = collider['lhcb1'].twiss()
twiss_b2 = collider['lhcb2'].twiss().reverse()
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
plt.plot(twiss_b1.s, twiss_b1.x, 'b')
plt.plot(twiss_b2.s, twiss_b2.x, 'r')

for ip in [1,2,5,8]:
    tw_b1_temp = twiss_b1[:, f'e.ds.l{ip}.*':f's.ds.r{ip}.*']
    tw_b2_temp = twiss_b2[:, f'e.ds.l{ip}.*':f's.ds.r{ip}.*']
    plt.figure()
    plt.subplot(211)
    plt.title(f'IP{ip}')
    plt.plot(tw_b1_temp.s-tw_b1_temp['s',f'ip{ip}'], tw_b1_temp.x, 'b')
    plt.plot(tw_b2_temp.s-tw_b2_temp['s',f'ip{ip}'], tw_b2_temp.x, 'r')
    plt.subplot(212)
    plt.plot(tw_b1_temp.s-tw_b1_temp['s',f'ip{ip}'], tw_b1_temp.y, 'b')
    plt.plot(tw_b2_temp.s-tw_b2_temp['s',f'ip{ip}'], tw_b2_temp.y, 'r')

# %%
plt.plot(tw_b1_temp.s-tw_b1_temp['s',f'ip{ip}'], tw_b1_temp.y, 'bo-')
plt.xlim(-25,25)

# %%
twiss_b1 = collider['lhcb1'].twiss()
twiss_b2 = collider['lhcb2'].twiss().reverse()

plt.plot(twiss_b1.s, twiss_b1.x, 'b')
plt.plot(twiss_b2.s, twiss_b2.x, 'r')

plt.figure()
plt.plot(twiss_b1.s, twiss_b1.y, 'b')
plt.plot(twiss_b2.s, twiss_b2.y, 'r')


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
# collider.vars['on_x2v'] = -350
# collider.vars['on_sep2h'] = .1401

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

from xtrack import lumi
# IP1 is levelled at pileup = 4
# IP2 is levelled at lumi = 1.25e31 cm-2s-1
# IP5 is head-on
# IP8 is levelled at pileup = 1 # decided a bit arbitrary
collider.vars['on_sep1'] = - 0.0215
collider.vars['on_sep2h'] = .164
collider.vars['on_sep8v'] = -0.055

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
        if ip == 'ip1':
            for param in ['on_x1', 'on_sep1', 'phi_ir1']:
                print(f'{param}: {collider.vars[param]._value:.4f}')
        
        if ip == 'ip5':
            for param in ['on_x5', 'on_sep5','phi_ir5']:
                print(f'{param}: {collider.vars[param]._value:.4f}')
        
        if ip == 'ip2':
            for param in ['on_x2h', 'on_x2v','on_sep2h', 'on_sep2v']:
                print(f'{param}: {collider.vars[param]._value:.4f}')
            print(f'ALICE polarity: {collider.vars["on_alice_normalized"]._value:.0f} ')
        if ip == 'ip8':
            for param in ['on_x8h', 'on_x8v','on_sep8h', 'on_sep8v']:
                print(f'{param}: {collider.vars[param]._value:.4f}')
            print(f'LHCb polarity: {collider.vars["on_lhcb_normalized"]._value:.0f}')

        print('\n')
    return my_luminosity, my_pileup

my_luminosity = {}
my_pileup = {}
for my_ip in (['ip1','ip2','ip5','ip8']):
    my_luminosity[my_ip], my_pileup[my_ip] = compute_luminosity(collider, ip=my_ip)

# %% Compute the footprint
collider.vars['beambeam_scale'] = 1
#collider['lhcb1'].vars['i_oct_b1'] = 100
fp1 = collider['lhcb1'].get_footprint(nemitt_x = collider.config['config_beambeam']['nemitt_x'],
                                      nemitt_y = collider.config['config_beambeam']['nemitt_y'],
                                      freeze_longitudinal=True, 
                                      n_turns=2048, 
                                      n_fft=2000000, 
                                      delta0=0, 
                                      zeta0=0)
# %%
fp1.plot(color='b', lw= 0, marker = 'o', markersize=2)




# %%

print('===== BB ON =====')
collider.vars['beambeam_scale'] = 1
print(collider['lhcb1'].twiss()[['betx','bety'],'ip[1,2,5,8]'])

print('\n===== BB OFF =====')
collider.vars['beambeam_scale'] = 0
print(collider['lhcb1'].twiss()[['betx','bety'],'ip[1,2,5,8]'])
collider.vars['beambeam_scale'] = 1

# %%
plt.plot(fp1.x_norm_2d, fp1.y_norm_2d, 'o')

# %%
plt.plot(fp1.qx)
# %%
plt.plot(fp1.qx, fp1.qy,'.')
# %%
import numpy as np

import xtrack as xt

class LinearRescale():

    def __init__(self, knob_name, v0, dv):
            self.knob_name = knob_name
            self.v0 = v0
            self.dv = dv

def _footprint_with_linear_rescale(linear_rescale_on_knobs, line,
                                   freeze_longitudinal=False,
                                   delta0=None, zeta0=None, kwargs={}):

        if isinstance (linear_rescale_on_knobs, LinearRescale):
            linear_rescale_on_knobs = [linear_rescale_on_knobs]

        assert len(linear_rescale_on_knobs) == 1, (
            'Only one linear rescale is supported for now')

        knobs_0 = {}
        for rr in linear_rescale_on_knobs:
            nn = rr.knob_name
            v0 = rr.v0
            knobs_0[nn] = v0

        with xt._temp_knobs(line, knobs_0):
            fp = line.get_footprint(
                freeze_longitudinal=freeze_longitudinal,
                delta0=delta0, zeta0=zeta0, **kwargs)

        qx0 = fp.qx
        qy0 = fp.qy

        for rr in linear_rescale_on_knobs:
            nn = rr.knob_name
            v0 = rr.v0
            dv = rr.dv

            knobs_1 = knobs_0.copy()
            knobs_1[nn] = v0 + dv

            with xt._temp_knobs(line, knobs_1):
                fp1 = get_footprint(line, freeze_longitudinal=freeze_longitudinal,
                                        delta0=delta0, zeta0=zeta0, **kwargs)
            delta_qx = (fp1.qx - qx0) / dv * (line.vars[nn]._value - v0)
            delta_qy = (fp1.qy - qy0) / dv * (line.vars[nn]._value - v0)

            fp.qx += delta_qx
            fp.qy += delta_qy
            fp.mon = fp1.mon

        return fp

class Footprint():

    def __init__(self, nemitt_x=None, nemitt_y=None, n_turns=256, n_fft=2**18,
            mode='polar', r_range=None, theta_range=None, n_r=None, n_theta=None,
            x_norm_range=None, y_norm_range=None, n_x_norm=None, n_y_norm=None,
            keep_fft=False):

        assert nemitt_x is not None and nemitt_y is not None, (
            'nemitt_x and nemitt_y must be provided')
        self.mode = mode

        self.n_turns = n_turns
        self.n_fft = n_fft
        self.keep_fft = keep_fft

        self.nemitt_x = nemitt_x
        self.nemitt_y = nemitt_y

        assert mode in ['polar', 'uniform_action_grid'], (
            'mode must be either polar or uniform_action_grid')

        if mode == 'polar':

            assert x_norm_range is None and y_norm_range is None, (
                'x_norm_range and y_norm_range must be None for mode polar')
            assert n_x_norm is None and n_y_norm is None, (
                'n_x_norm and n_y_norm must be None for mode polar')

            if r_range is None:
                r_range = (0.1, 6)
            if theta_range is None:
                theta_range = (0.05, np.pi/2-0.05)
            if n_r is None:
                n_r = 10
            if n_theta is None:
                n_theta = 10

            self.r_range = r_range
            self.theta_range = theta_range
            self.n_r = n_r
            self.n_theta = n_theta

            self.r_grid = np.linspace(*r_range, n_r)
            self.theta_grid = np.linspace(*theta_range, n_theta)
            self.R_2d, self.Theta_2d = np.meshgrid(self.r_grid, self.theta_grid)

            self.x_norm_2d = self.R_2d * np.cos(self.Theta_2d)
            self.y_norm_2d = self.R_2d * np.sin(self.Theta_2d)

        elif mode == 'uniform_action_grid':

            assert r_range is None and theta_range is None, (
                'r_range and theta_range must be None for mode uniform_action_grid')
            assert n_r is None and n_theta is None, (
                'n_r and n_theta must be None for mode uniform_action_grid')

            if x_norm_range is None:
                x_norm_range = (0.1, 6)
            if y_norm_range is None:
                y_norm_range = (0.1, 6)
            if n_x_norm is None:
                n_x_norm = 10
            if n_y_norm is None:
                n_y_norm = 10

            Jx_min = nemitt_x * x_norm_range[0]**2 / 2
            Jx_max = nemitt_x * x_norm_range[1]**2 / 2
            Jy_min = nemitt_y * y_norm_range[0]**2 / 2
            Jy_max = nemitt_y * y_norm_range[1]**2 / 2

            self.Jx_grid = np.linspace(Jx_min, Jx_max, n_x_norm)
            self.Jy_grid = np.linspace(Jy_min, Jy_max, n_y_norm)

            self.Jx_2d, self.Jy_2d = np.meshgrid(self.Jx_grid, self.Jy_grid)

            self.x_norm_2d = np.sqrt(2 * self.Jx_2d / nemitt_x)
            self.y_norm_2d = np.sqrt(2 * self.Jy_2d / nemitt_y)

    def _compute_footprint(self, line, freeze_longitudinal=False,
                           delta0=None, zeta0=None):

        if freeze_longitudinal is None:
            # In future we could detect if the line has frozen longitudinal plane
            freeze_longitudinal = False

        particles = line.build_particles(
            x_norm=self.x_norm_2d.flatten(), y_norm=self.y_norm_2d.flatten(),
            nemitt_x=self.nemitt_x, nemitt_y=self.nemitt_y,
            zeta=zeta0, delta=delta0,
            freeze_longitudinal=freeze_longitudinal,
            method={True: '4d', False: '6d'}[freeze_longitudinal]
            )

        print('Tracking particles for footprint...')
        line.track(particles, num_turns=self.n_turns, turn_by_turn_monitor=True,
                   freeze_longitudinal=freeze_longitudinal)
        print('Done tracking.')

        ctx2np = line._context.nparray_from_context_array
        assert np.all(ctx2np(particles.state == 1)), (
            'Some particles were lost during tracking')
        mon = line.record_last_track

        print('Computing footprint...')
        fft_x = np.fft.rfft(
            mon.x - np.atleast_2d(np.mean(mon.x, axis=1)).T, n=self.n_fft, axis=1)
        fft_y = np.fft.rfft(
            mon.y - np.atleast_2d(np.mean(mon.y, axis=1)).T, n=self.n_fft, axis=1)

        if self.keep_fft:
            self.fft_x = fft_x
            self.fft_y = fft_y

        freq_axis = np.fft.rfftfreq(self.n_fft)

        qx = freq_axis[np.argmax(np.abs(fft_x), axis=1)]
        qy = freq_axis[np.argmax(np.abs(fft_y), axis=1)]

        self.qx = np.reshape(qx, self.x_norm_2d.shape)
        self.qy = np.reshape(qy, self.x_norm_2d.shape)
        self.mon = mon
        print ('Done computing footprint.')

    def plot(line, ax=None, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        if 'color' not in kwargs:
            kwargs['color'] = 'k'

        labels = [None] * line.qx.shape[1]

        if 'label' in kwargs:
            label_str = kwargs['label']
            kwargs.pop('label')
            labels[0] = label_str

        ax.plot(line.qx, line.qy, label=labels, **kwargs)
        ax.plot(line.qx.T, line.qy.T, **kwargs)

        ax.set_xlabel(r'$q_x$')
        ax.set_ylabel(r'$q_y$')
        
def get_footprint(line, nemitt_x=None, nemitt_y=None, n_turns=256, n_fft=2**18,
        mode='polar', r_range=None, theta_range=None, n_r=None, n_theta=None,
        x_norm_range=None, y_norm_range=None, n_x_norm=None, n_y_norm=None,
        linear_rescale_on_knobs=None,
        freeze_longitudinal=None, delta0=None, zeta0=None,
        keep_fft=True):

    '''
    Compute the tune footprint for a beam with given emittences using tracking.

    Parameters
    ----------

    nemitt_x : float
        Normalized emittance in the x-plane.
    nemitt_y : float
        Normalized emittance in the y-plane.
    n_turns : int
        Number of turns for tracking.
    n_fft : int
        Number of points for FFT (tracking data is zero-padded to this length).
    mode : str
        Mode for computing footprint. Options are 'polar' and 'uniform_action_grid'.
        In 'polar' mode, the footprint is computed on a polar grid with
        r_range and theta_range specifying the range of r and theta values (
        polar coordinates in the x_norm, y_norm plane).
        In 'uniform_action_grid' mode, the footprint is computed on a uniform
        grid in the action space (Jx, Jy).
    r_range : tuple of floats
        Range of r values for footprint in polar mode. Default is (0.1, 6) sigmas.
    theta_range : tuple of floats
        Range of theta values for footprint in polar mode. Default is
        (0.05, pi / 2 - 0.05) radians.
    n_r : int
        Number of r values for footprint in polar mode. Default is 10.
    n_theta : int
        Number of theta values for footprint in polar mode. Default is 10.
    x_norm_range : tuple of floats
        Range of x_norm values for footprint in `uniform action grid` mode.
        Default is (0.1, 6) sigmas.
    y_norm_range : tuple of floats
        Range of y_norm values for footprint in `uniform action grid` mode.
        Default is (0.1, 6) sigmas.
    n_x_norm : int
        Number of x_norm values for footprint in `uniform action grid` mode.
        Default is 10.
    n_y_norm : int
        Number of y_norm values for footprint in `uniform action grid` mode.
        Default is 10.
    linear_rescale_on_knobs: list of xt.LinearRescale
        Detuning from listed knobs is evaluated at a given value of the knob
        with the provided step and rescaled to the actual knob value.
        This is useful to avoid artefact from linear coupling or resonances.
        Example:
            ``line.get_footprint(..., linear_rescale_on_knobs=[
                xt.LinearRescale(knob_name='beambeam_scale', v0=0, dv-0.1)])``
    freeze_longitudinal : bool
        If True, the longitudinal coordinates are frozen during the particles
        matching and the tracking.
    delta0: float
        Initial value of the delta coordinate.
    zeta0: float
        Initial value of the zeta coordinate.

    Returns
    -------
    fp : Footprint
        Footprint object containing footprint data (fp.qx, fp.qy).

    '''

    kwargs = locals()
    kwargs.pop('line')
    kwargs.pop('linear_rescale_on_knobs')

    freeze_longitudinal = kwargs.pop('freeze_longitudinal')
    delta0 = kwargs.pop('delta0')
    zeta0 = kwargs.pop('zeta0')

    if linear_rescale_on_knobs:
        fp = _footprint_with_linear_rescale(line=line, kwargs=kwargs,
                    linear_rescale_on_knobs=linear_rescale_on_knobs,
                    freeze_longitudinal=freeze_longitudinal,
                    delta0=delta0, zeta0=zeta0)
    else:
        fp = Footprint(**kwargs)
        fp._compute_footprint(line,
            freeze_longitudinal=freeze_longitudinal,
            delta0=delta0, zeta0=zeta0)

    return fp

# %%
#collider.vars['beambeam_scale'] = 1
#collider['lhcb1'].vars['i_oct_b1'] = 100
#collider.vars['dqy.b1_sq'] -= 0.006

fp1 = get_footprint(collider['lhcb1'],
                    nemitt_x = collider.config['config_beambeam']['nemitt_x'],
                    nemitt_y = collider.config['config_beambeam']['nemitt_y'],
                    freeze_longitudinal=False, 
                    theta_range = (0.05, np.pi / 2 - 0.05),
                    n_r =11,
                    n_theta=11,
                    n_turns=1024, 
                    n_fft=1024000, 
                    delta0=0, 
                    zeta0=0,
                    mode='polar',)
# %%
import PyNAFF as pnf
import NAFFlib
from PySUSSIX import PySussix

my_twiss = collider['lhcb1'].twiss()

signal = fp1.mon.x[-10,:]
signal = fp1.mon.y[-10,:]
qx_pynaff = []
qy_pynaff = []
qx_nafflib = []
qy_nafflib = []
qx_sussix = []
qy_sussix = []


(my_len,_) = np.shape(fp1.mon.x)

for ii in range(my_len):   
    signal = fp1.mon.x[ii,:]
    # qx_pynaff.append(pnf.naff(signal, fp1.n_turns, 1, 0 , False)[0][1])
    qx_nafflib.append(NAFFlib.get_tune(signal))
    signal = fp1.mon.y[ii,:]
    # qy_pynaff.append(pnf.naff(signal, fp1.n_turns, 1, 0 , False)[0][1])
    qy_nafflib.append(NAFFlib.get_tune(signal))

    my_sussix = PySussix.Sussix()
    my_sussix.sussix_inp(nt1=1, nt2=fp1.n_turns, idam=2, ir=1,
                   tunex=my_twiss.qx-np.floor(my_twiss.qx), 
                   tuney=my_twiss.qy-np.floor(my_twiss.qy))
    
    ans  = my_sussix.sussix(fp1.mon.x[ii,:],
                     fp1.mon.px[ii,:],
                     fp1.mon.y[ii,:],
                     fp1.mon.py[ii,:],
                     fp1.mon.zeta[ii,:],
                     fp1.mon.delta[ii,:])
    qx_sussix.append(my_sussix.ox[0])
    qy_sussix.append(my_sussix.oy[0])

# %%
fp1.plot(color='b',linewidth=0, marker='o', markersize=3)

plt.plot(qx_nafflib, qy_nafflib,'+r')
#plt.plot(qx_pynaff, qy_pynaff,'.k')
plt.plot(qx_sussix, qy_sussix,'xm')

plt.plot([.29,.32],[.29,.32],'-')
# plt.xlim(.305,.315)
# plt.ylim(.315,.325)

# %%

plt.plot(fp1.x_norm_2d, fp1.y_norm_2d, 'ro')

aux = np.reshape(np.array(qx_sussix)-np.array(qy_sussix), fp1.x_norm_2d.shape)
my_filter = np.abs(aux)<1e-4
plt.plot(fp1.x_norm_2d[my_filter], fp1.y_norm_2d[my_filter], 'b.')



# %%
# importing matplot lib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d as interp1d

# importing movie py libraries
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
 
 
# duration of the video
duration = 10
my_fps = 1
t = np.linspace(0, duration, duration+1)
on_x2v_range = np.linspace(-170, -350, duration +1)
on_sep2h_range = np.linspace(-3.5, 0, duration +1)

on_x2v_linear = interp1d(t, on_x2v_range) 
on_sep2h_linear = interp1d(t, on_sep2h_range)
collider.vars["on_alice_normalized"] = 1

 
# matplot subplot
fig, ax = plt.subplots()
 
# method to get frames
def make_frame(t):
     
    # clear
    ax.clear()
    on_x2v =on_x2v_linear(t)
    on_sep2h = on_sep2h_linear(t) 
    collider.vars['on_x2v'] = on_x2v
    collider.vars['on_sep2h'] = on_sep2h
    collider.configure_beambeam_interactions(config_collider['config_beambeam']['num_particles_per_bunch'],
                                             config_collider['config_beambeam']['nemitt_x'],
                                             config_collider['config_beambeam']['nemitt_y'],
                                             crab_strong_beam=False,)
    collider.vars['beambeam_scale'] = 1

    my_dict = compute_separation(collider, ip='ip2')
    ip_dict = my_dict
    ax.plot(ip_dict['s'], np.abs(ip_dict['dx_sig']), 'ob', label='x')
    ax.plot(ip_dict['s'], np.abs(ip_dict['dy_sig']), 'sr', label='y')
    ax.set_xlabel('s [m]')
    ax.set_ylabel('separation [$\sigma$]')
    ax.legend(loc='upper right')
    ax.grid(True)
    # plot_orbits(my_dict)
    # ax = plt.gca()
    # ax.set_title(f'on_x2v={on_x2v}, on_sep2h={on_sep2h}')
    # plot_separation(my_dict)
    # ax = plt.gca()
    ax.set_ylim(8, 10)
    ax.set_title(f'on_x2v={on_x2v:.1f}, on_sep2h={on_sep2h:.2f}, ALICE polarity = {collider.vars["on_alice_normalized"]._value}')
    # returning numpy image
    return mplfig_to_npimage(fig)
 
# creating animation
animation = VideoClip(make_frame, duration = duration+1)
 
# displaying animation with auto play and looping
animation.ipython_display(fps = 1, loop = True, autoplay = True)
# %%

fig, ax = plt.subplots()
 
# method to get frames
def make_frame(t):
     
    # clear
    ax.clear()
    on_x2v =on_x2v_linear(t)
    on_sep2h = on_sep2h_linear(t) 
    collider.vars['on_x2v'] = on_x2v
    collider.vars['on_sep2h'] = on_sep2h
    my_dict = compute_separation(collider, ip='ip2')
    ip_dict = my_dict

    ax.plot(my_dict['twiss_filtered']['b1']['s'], 
            my_dict['twiss_filtered']['b1']['x']
            +my_dict['survey_filtered']['b1']['X'],
            'ob-', label='B1 x'
    )
    ax.plot(my_dict['twiss_filtered']['b2']['s'], 
            my_dict['twiss_filtered']['b2']['x']
            +my_dict['survey_filtered']['b2']['X'],
            'or-', label='B2 x'
    )

    ax.plot(my_dict['twiss_filtered']['b1']['s'], 
            my_dict['twiss_filtered']['b1']['y'],
            'ob:', label='B1 y'
    )
    ax.plot(my_dict['twiss_filtered']['b2']['s'], 
            my_dict['twiss_filtered']['b2']['y'],
            'or:', label='B2 y'
    )


    ax.set_xlabel('s [m]')
    ax.set_ylabel('separation [$\sigma$]')
    ax.legend(loc='upper right')
    ax.grid(True)

    # plot_orbits(my_dict)
    # ax = plt.gca()
    # ax.set_title(f'on_x2v={on_x2v}, on_sep2h={on_sep2h}')
    # plot_separation(my_dict)
    # ax = plt.gca()
    ax.set_title(f'on_x2v={on_x2v:.1f}, on_sep2h={on_sep2h:.2f}, ALICE polarity = {collider.vars["on_alice_normalized"]._value}')
    # returning numpy image
    return mplfig_to_npimage(fig)
 
# creating animation
animation = VideoClip(make_frame, duration = duration+1)
 
# displaying animation with auto play and looping
animation.ipython_display(fps = 1, loop = True, autoplay = True)

# %%
