# Imports
import json
import logging
from scipy.constants import c as clight
from scipy.optimize import minimize_scalar
import sys
sys.path.insert(0,"/afs/cern.ch/work/s/skostogl/private/workspaces_2023/xtrack/xtrack")
import xtrack as xt
import numpy as np
import xmask.lhc as xlhc
import pandas as pd

def user_function_after_loading_collider(collider):
  # Save twisses, orbit should be flat at this stage
  return
  for beam in ['lhcb1', 'lhcb2']:
    twiss = collider[beam].twiss().to_pandas()
    twiss.drop(columns='W_matrix').to_parquet(f"twiss_{beam}_crossing_disabled.parquet")
  return

def user_function_after_knobs(collider):
  return
  # Orbit plots with and without crossing and sanity checks of internal external crossing angles

  import matplotlib.pyplot as plt
  fig, ax = plt.subplots(ncols=2, nrows=2)
  
  twisses = {} 
  for axes,beam in enumerate(['lhcb1', 'lhcb2']):
    twisses[beam] = {}
    twiss = collider[beam].twiss().to_pandas()
    twisses[beam]["xing"] = twiss
    
    plt.sca(ax[axes, 0])
    plt.xlabel("s (m)")
    plt.ylabel("Orbit (m)")
    plt.title(f"{beam}, crossing enabled")
    plt.plot(twiss.s, twiss.x, lw=2, label='x')
    plt.plot(twiss.s, twiss.y, lw=2, label='y')
    plt.legend()
    
    twiss.drop(columns='W_matrix').to_parquet(f"twiss_{beam}_crossing_enabled.parquet")
    twiss = pd.read_parquet(f"twiss_{beam}_crossing_disabled.parquet")
    twisses[beam]["noxing"] = twiss
    
    plt.sca(ax[axes, 1])
    plt.title(f"{beam}, crossing disabled")
    plt.plot(twiss.s, twiss.x, lw=2)
    plt.plot(twiss.s, twiss.y, lw=2)
  fig.tight_layout() 
  fig.savefig("twiss.png")

  twiss = collider['lhcb1'].twiss().to_pandas()
  for ip in [1,5, 2, 8]: 
    print(f"IP{ip}: from twiss ",np.sqrt((twiss[twiss.name == f'ip{ip}'].px.values[0]*1e6)**2 + (twiss[twiss.name == f'ip{ip}'].py.values[0]*1e6)**2), " from config ", collider.vars[f"on_x{ip}"]._value)
  
  # Internal crossing * polarity = Total-external
  print("Internal IP2:, ",(twiss[twiss.name == f'ip2'].py.values[0]*1e6 - collider.vars[f"on_x2"]._value))
  print("Internal IP8:, ",(twiss[twiss.name == f'ip8'].px.values[0]*1e6 - collider.vars[f"on_x8"]._value))
     
  return

def user_function_after_matching(collider):
  return

def do_levelling(
            config_collider,
            config_bb,
            n_collisions_ip2,
            n_collisions_ip8,
            collider,
            n_collisions_ip1_and_5,
            crab=False
            ):
    # Read knobs and tuning settings from config file (already updated with the number of collisions)
    config_lumi_leveling = config_collider["config_lumi_leveling"]

    # Update the number of bunches in the configuration file
    config_lumi_leveling["ip1"]["num_colliding_bunches"] = int(n_collisions_ip1_and_5)
    config_lumi_leveling["ip5"]["num_colliding_bunches"] = int(n_collisions_ip1_and_5)
    config_lumi_leveling["ip2"]["num_colliding_bunches"] = int(n_collisions_ip2)
    config_lumi_leveling["ip8"]["num_colliding_bunches"] = int(n_collisions_ip8)

    # Level by separation
    try:
        xlhc.luminosity_leveling(
        collider, config_lumi_leveling=config_lumi_leveling,
        config_beambeam=config_bb)
    except:
        print("Leveling failed..continuing")
    
    l_n_collisions = [n_collisions_ip1_and_5, n_collisions_ip2, n_collisions_ip1_and_5, n_collisions_ip8]
    config_bb = record_final_luminosity(collider, config_bb, l_n_collisions, crab)

    lumis_tot = []
    seps_tot = []
    for ip in list(config_lumi_leveling.keys()):
      lumi = config_bb[f"luminosity_{ip}_after_optimization"]
      lumis_tot.append(lumi)
      for knob in  config_lumi_leveling[ip]["knobs"]:
        config_lumi_leveling[ip][f"final_{knob}"] = float(
        collider.vars[knob]._value
        )

      on_sep = collider.vars[f"on_sep{ip[-1]}"]._value
      lumi = config_bb[f"luminosity_{ip}_after_optimization"]
      #lumis_tot.append(lumi) 
      print(f"IP{ip[-1]} separation: {on_sep}, lumi: {lumi}")
    onx1 = collider.vars["on_x1"]._value
    onx2 = collider.vars["on_x2"]._value
    onx5 = collider.vars["on_x5"]._value
    onx8 = collider.vars["on_x8"]._value
    on_alice = collider.vars["on_alice"]._value
    on_lhcb = collider.vars["on_lhcb"]._value

    path_to_save = "/eos/user/s/skostogl/workspaces_2023/Ions/TCT_twiss"
    pd.DataFrame({"lumi": lumis_tot, "ip":list(config_lumi_leveling.keys())}).to_parquet(f"{path_to_save}/LUMI_onx1{onx1}_onx2{onx2}_onx8{onx8}_onlhc{on_lhcb}_onalice{on_alice}.parquet")
    return collider, config_collider, crab

def user_function_after_leveling(collider):
  return


# ==================================================================================================
# --- Function to compute luminosity once the collider is configured
# ==================================================================================================
def record_final_luminosity(collider, config_bb, l_n_collisions, crab):

    # Get the final luminoisty in all IPs
    twiss_b1 = collider["lhcb1"].twiss()
    twiss_b2 = collider["lhcb2"].twiss()
    l_lumi = []
    l_PU = []
    l_ip = ["ip1", "ip2", "ip5", "ip8"]
    for n_col, ip in zip(l_n_collisions, l_ip):
        try:
            L = xt.lumi.luminosity_from_twiss(
                    n_colliding_bunches=n_col,
                    num_particles_per_bunch=config_bb["num_particles_per_bunch"],
                    ip_name=ip,
                    nemitt_x=config_bb["nemitt_x"],
                    nemitt_y=config_bb["nemitt_y"],
                    sigma_z=config_bb["sigma_z"],
                    twiss_b1=twiss_b1,
                    twiss_b2=twiss_b2,
                    crab=crab,
                )
            PU = compute_PU(L, n_col, twiss_b1["T_rev0"])
        except Exception as e:
            print(f"There was a problem during the luminosity computation in {ip}... Ignoring it. Error: {e}")
            L = 0
            PU = 0
        l_lumi.append(L)
        l_PU.append(PU)

    # Update configuration
    for ip, L, PU in zip(l_ip, l_lumi, l_PU):
        config_bb[f"luminosity_{ip}_after_optimization"] = float(L)
        config_bb[f"Pile-up_{ip}_after_optimization"] = float(PU)

    return config_bb


def user_function_after_coupling(collider):
  return

def user_function_before_bb(collider):
  return
  # plotting footprints

  import matplotlib.pyplot as plt

  config, config_mad = read_configuration("config.yaml")
  config_bb = config["config_collider"]["config_beambeam"]

  fig, ax = plt.subplots(ncols=2, figsize=(12,6))
  plt.sca(ax[0])

  fp = collider['lhcb1'].get_footprint(
    nemitt_x=config_bb["nemitt_x"], nemitt_y=config_bb["nemitt_y"], r_range=[0.1, 10], freeze_longitudinal=True, delta0=0)
  fp.plot(label='dp/p=0')
  fp = collider['lhcb1'].get_footprint(
    nemitt_x=config_bb["nemitt_x"], nemitt_y=config_bb["nemitt_y"], r_range=[0.1, 10], freeze_longitudinal=True, delta0=config["config_simulation"]["delta_max"])
  fp.plot(color='r', label='with dp/p')
 
  qx = config["config_collider"]["config_knobs_and_tuning"]["qx"]["lhcb1"]
  qy = config["config_collider"]["config_knobs_and_tuning"]["qy"]["lhcb1"]
  qx = qx - int(qx)
  qy = qy - int(qy)
  plt.scatter(qx, qy, marker='*', s=200)
  plt.legend()

  plt.sca(ax[1])
  x = fp.r_grid*np.cos(fp.Theta_2d)
  y = fp.r_grid*np.sin(fp.Theta_2d)
  plt.plot(x,y, '-k'); plt.plot(x.T, y.T, '.-k')
  plt.xlabel("x (sigma)")
  plt.ylabel("y (sigma)")
  fig.tight_layout()
  plt.savefig("footprint_withoutBB.png")
  
  return

def user_function_after_bb(collider):
  # plotting footprints
  return

  import matplotlib.pyplot as plt

  config, config_mad = read_configuration("config.yaml")
  config_bb = config["config_collider"]["config_beambeam"]

  fig, ax = plt.subplots(ncols=2, figsize=(12,6))
  plt.sca(ax[0])

  fp = collider['lhcb1'].get_footprint(
    nemitt_x=config_bb["nemitt_x"], nemitt_y=config_bb["nemitt_y"], r_range=[0.1, 10], freeze_longitudinal=True, delta0=0)
  fp.plot(label='dp/p=0')
  fp = collider['lhcb1'].get_footprint(
    nemitt_x=config_bb["nemitt_x"], nemitt_y=config_bb["nemitt_y"], r_range=[0.1, 10], freeze_longitudinal=True, delta0=config["config_simulation"]["delta_max"])
  fp.plot(color='r', label='with dp/p')
 
  qx = config["config_collider"]["config_knobs_and_tuning"]["qx"]["lhcb1"]
  qy = config["config_collider"]["config_knobs_and_tuning"]["qy"]["lhcb1"]
  qx = qx - int(qx)
  qy = qy - int(qy)
  plt.scatter(qx, qy, marker='*', s=200)
  plt.legend()

  plt.sca(ax[1])
  x = fp.r_grid*np.cos(fp.Theta_2d)
  y = fp.r_grid*np.sin(fp.Theta_2d)
  plt.plot(x,y, '-k'); plt.plot(x.T, y.T, '.-k')
  plt.xlabel("x (sigma)")
  plt.ylabel("y (sigma)")
  fig.tight_layout()
  plt.savefig("footprint_withBB.png")

  # switch off long-ranges, octupoles, crossing angle and keep only IP1 HO
  collider_copy = xt.Multiline.from_dict(collider.to_dict())
  for ip in [1,2,5,8]:
    collider_copy = disable_bb(collider_copy, ip, 'lr') 
  for ip in [2,5,8]:
    collider_copy = disable_bb(collider_copy, ip, 'ho') 
  
  collider_copy.vars["i_oct_b1"] = 0
  collider_copy.vars["i_oct_b2"] = 0
  for ip in [1,2,5,8]:
    collider_copy.vars[f"on_x{ip}"] = 0
    collider_copy.vars[f"on_sep{ip}"] = 0

  collider_copy.build_trackers()

  fig, ax = plt.subplots(ncols=2, figsize=(12,6))
  plt.sca(ax[0])

  fp = collider_copy['lhcb1'].get_footprint(
    nemitt_x=config_bb["nemitt_x"], nemitt_y=config_bb["nemitt_y"], r_range=[0.1, 6], freeze_longitudinal=True, delta0=0)
  fp.plot(label='dp/p=0')
 
  qx = config["config_collider"]["config_knobs_and_tuning"]["qx"]["lhcb1"]
  qy = config["config_collider"]["config_knobs_and_tuning"]["qy"]["lhcb1"]
  qx = qx - int(qx)
  qy = qy - int(qy)
  plt.scatter(qx, qy, marker='*', s=200)
  plt.legend()

  plt.sca(ax[1])
  x = fp.r_grid*np.cos(fp.Theta_2d)
  y = fp.r_grid*np.sin(fp.Theta_2d)
  plt.plot(x,y, '-k'); plt.plot(x.T, y.T, '.-k')
  plt.xlabel("x (sigma)")
  plt.ylabel("y (sigma)")
  fig.tight_layout()
  plt.savefig("footprint_HOIP1.png")

  return

def disable_bb(collider_copy, ip, ho_or_lr='lr'):
  if ho_or_lr not in ["ho", "lr"]:
    print("ho or lr must be specified, continuing without de-activating any BB lens")
    return
  for i in collider_copy.vars.keys():
    if ('scale_strength' in i) and (f"bb_{ho_or_lr}" in i) and (f'{ip}b' in i):
      collider_copy.vars[i] = 0
  return collider_copy


def get_fma(collider):
    return

def user_function_after_final_collider_configuration(collider):
  #get_fma()

  for beam in ['lhcb1', 'lhcb2']:
    twiss = collider[beam].twiss().to_pandas()
    onx1 = collider.vars["on_x1"]._value
    onx2 = collider.vars["on_x2"]._value
    onx5 = collider.vars["on_x5"]._value
    onx8 = collider.vars["on_x8"]._value
    on_alice = collider.vars["on_alice"]._value
    on_lhcb = collider.vars["on_lhcb"]._value
    path_to_save = "/eos/user/s/skostogl/workspaces_2023/Ions/TCT_twiss"
    twiss.drop(columns='W_matrix').to_parquet(f"{path_to_save}/twiss_{beam}_final_onx1{onx1}_onx2{onx2}_onx8{onx8}_onlhc{on_lhcb}_onalice{on_alice}.parquet")
  return

def user_function_after_tracking(particles):
  return

# Function to generate dictionnary containing the orbit correction setup
def generate_orbit_correction_setup():
    correction_setup = {}
    correction_setup["lhcb1"] = {
        "IR1 left": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.r8.b1",
            end="e.ds.l1.b1",
            vary=(
                "corr_co_acbh14.l1b1",
                "corr_co_acbh12.l1b1",
                "corr_co_acbv15.l1b1",
                "corr_co_acbv13.l1b1",
            ),
            targets=("e.ds.l1.b1",),
        ),
        "IR1 right": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.r1.b1",
            end="s.ds.l2.b1",
            vary=(
                "corr_co_acbh13.r1b1",
                "corr_co_acbh15.r1b1",
                "corr_co_acbv12.r1b1",
                "corr_co_acbv14.r1b1",
            ),
            targets=("s.ds.l2.b1",),
        ),
        "IR5 left": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.r4.b1",
            end="e.ds.l5.b1",
            vary=(
                "corr_co_acbh14.l5b1",
                "corr_co_acbh12.l5b1",
                "corr_co_acbv15.l5b1",
                "corr_co_acbv13.l5b1",
            ),
            targets=("e.ds.l5.b1",),
        ),
        "IR5 right": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.r5.b1",
            end="s.ds.l6.b1",
            vary=(
                "corr_co_acbh13.r5b1",
                "corr_co_acbh15.r5b1",
                "corr_co_acbv12.r5b1",
                "corr_co_acbv14.r5b1",
            ),
            targets=("s.ds.l6.b1",),
        ),
        "IP1": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.l1.b1",
            end="s.ds.r1.b1",
            vary=(
                "corr_co_acbch6.l1b1",
                "corr_co_acbcv5.l1b1",
                "corr_co_acbch5.r1b1",
                "corr_co_acbcv6.r1b1",
                "corr_co_acbyhs4.l1b1",
                "corr_co_acbyhs4.r1b1",
                "corr_co_acbyvs4.l1b1",
                "corr_co_acbyvs4.r1b1",
            ),
            targets=("ip1", "s.ds.r1.b1"),
        ),
        "IP2": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.l2.b1",
            end="s.ds.r2.b1",
            vary=(
                "corr_co_acbyhs5.l2b1",
                "corr_co_acbchs5.r2b1",
                "corr_co_acbyvs5.l2b1",
                "corr_co_acbcvs5.r2b1",
                "corr_co_acbyhs4.l2b1",
                "corr_co_acbyhs4.r2b1",
                "corr_co_acbyvs4.l2b1",
                "corr_co_acbyvs4.r2b1",
            ),
            targets=("ip2", "s.ds.r2.b1"),
        ),
        "IP5": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.l5.b1",
            end="s.ds.r5.b1",
            vary=(
                "corr_co_acbch6.l5b1",
                "corr_co_acbcv5.l5b1",
                "corr_co_acbch5.r5b1",
                "corr_co_acbcv6.r5b1",
                "corr_co_acbyhs4.l5b1",
                "corr_co_acbyhs4.r5b1",
                "corr_co_acbyvs4.l5b1",
                "corr_co_acbyvs4.r5b1",
            ),
            targets=("ip5", "s.ds.r5.b1"),
        ),
        "IP8": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.l8.b1",
            end="s.ds.r8.b1",
            vary=(
                "corr_co_acbch5.l8b1",
                "corr_co_acbyhs4.l8b1",
                "corr_co_acbyhs4.r8b1",
                "corr_co_acbyhs5.r8b1",
                "corr_co_acbcvs5.l8b1",
                "corr_co_acbyvs4.l8b1",
                "corr_co_acbyvs4.r8b1",
                "corr_co_acbyvs5.r8b1",
            ),
            targets=("ip8", "s.ds.r8.b1"),
        ),
    }

    correction_setup["lhcb2"] = {
        "IR1 left": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.l1.b2",
            end="e.ds.r8.b2",
            vary=(
                "corr_co_acbh13.l1b2",
                "corr_co_acbh15.l1b2",
                "corr_co_acbv12.l1b2",
                "corr_co_acbv14.l1b2",
            ),
            targets=("e.ds.r8.b2",),
        ),
        "IR1 right": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.l2.b2",
            end="s.ds.r1.b2",
            vary=(
                "corr_co_acbh12.r1b2",
                "corr_co_acbh14.r1b2",
                "corr_co_acbv13.r1b2",
                "corr_co_acbv15.r1b2",
            ),
            targets=("s.ds.r1.b2",),
        ),
        "IR5 left": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.l5.b2",
            end="e.ds.r4.b2",
            vary=(
                "corr_co_acbh13.l5b2",
                "corr_co_acbh15.l5b2",
                "corr_co_acbv12.l5b2",
                "corr_co_acbv14.l5b2",
            ),
            targets=("e.ds.r4.b2",),
        ),
        "IR5 right": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.l6.b2",
            end="s.ds.r5.b2",
            vary=(
                "corr_co_acbh12.r5b2",
                "corr_co_acbh14.r5b2",
                "corr_co_acbv13.r5b2",
                "corr_co_acbv15.r5b2",
            ),
            targets=("s.ds.r5.b2",),
        ),
        "IP1": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.r1.b2",
            end="e.ds.l1.b2",
            vary=(
                "corr_co_acbch6.r1b2",
                "corr_co_acbcv5.r1b2",
                "corr_co_acbch5.l1b2",
                "corr_co_acbcv6.l1b2",
                "corr_co_acbyhs4.l1b2",
                "corr_co_acbyhs4.r1b2",
                "corr_co_acbyvs4.l1b2",
                "corr_co_acbyvs4.r1b2",
            ),
            targets=(
                "ip1",
                "e.ds.l1.b2",
            ),
        ),
        "IP2": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.r2.b2",
            end="e.ds.l2.b2",
            vary=(
                "corr_co_acbyhs5.l2b2",
                "corr_co_acbchs5.r2b2",
                "corr_co_acbyvs5.l2b2",
                "corr_co_acbcvs5.r2b2",
                "corr_co_acbyhs4.l2b2",
                "corr_co_acbyhs4.r2b2",
                "corr_co_acbyvs4.l2b2",
                "corr_co_acbyvs4.r2b2",
            ),
            targets=("ip2", "e.ds.l2.b2"),
        ),
        "IP5": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.r5.b2",
            end="e.ds.l5.b2",
            vary=(
                "corr_co_acbch6.r5b2",
                "corr_co_acbcv5.r5b2",
                "corr_co_acbch5.l5b2",
                "corr_co_acbcv6.l5b2",
                "corr_co_acbyhs4.l5b2",
                "corr_co_acbyhs4.r5b2",
                "corr_co_acbyvs4.l5b2",
                "corr_co_acbyvs4.r5b2",
            ),
            targets=(
                "ip5",
                "e.ds.l5.b2",
            ),
        ),
        "IP8": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.r8.b2",
            end="e.ds.l8.b2",
            vary=(
                "corr_co_acbchs5.l8b2",
                "corr_co_acbyhs5.r8b2",
                "corr_co_acbcvs5.l8b2",
                "corr_co_acbyvs5.r8b2",
                "corr_co_acbyhs4.l8b2",
                "corr_co_acbyhs4.r8b2",
                "corr_co_acbyvs4.l8b2",
                "corr_co_acbyvs4.r8b2",
            ),
            targets=(
                "ip8",
                "e.ds.l8.b2",
            ),
        ),
    }
    return correction_setup

def compute_PU(luminosity, num_colliding_bunches, T_rev0, cross_section = 81e-27):
    return (
            luminosity
            / num_colliding_bunches
            * cross_section
            * T_rev0
        )

# ==================================================================================================
# --- Functions to read configuration files and generate configuration files for orbit correction
# ==================================================================================================
def read_configuration(config_path="config.yaml"):
    import ruamel.yaml
    ryaml = ruamel.yaml.YAML()
    # Read configuration for simulations
    with open(config_path, "r") as fid:
        config = ryaml.load(fid)

    # Also read configuration from previous generation
    try:
        with open("../" + config_path, "r") as fid:
            config_gen_1 = ryaml.load(fid)
    except:
        with open("../1_build_distr_and_collider/" + config_path, "r") as fid:
            config_gen_1 = ryaml.load(fid)

    config_mad = config_gen_1["config_mad"]
    return config, config_mad


if __name__ == "__main__":
    correction_setup = generate_orbit_correction_setup()
    for nn in ["lhcb1", "lhcb2"]:
        with open(f"corr_co_{nn}.json", "w") as fid:
            json.dump(correction_setup[nn], fid, indent=4)
