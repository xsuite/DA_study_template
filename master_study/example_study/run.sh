#!/bin/bash
source /afs/cern.ch/work/c/cdroin/private/example_DA_study/master_study/../miniconda/bin/activate
cd /afs/cern.ch/work/c/cdroin/private/example_DA_study/master_study/example_study
python 1_build_distr_and_collider.py > output.txt 2> error.txt
rm -rf final_* modules optics_repository optics_toolkit tools tracking_tools temp mad_collider.log __pycache__ twiss* errors fc* optics_orbit_at*