#!/bin/bash
source /afs/cern.ch/work/c/cdroin/private/example_DA_study_runIII_ions/master_study/../activate_miniforge.sh
cp -f /afs/cern.ch/work/c/cdroin/private/example_DA_study_runIII_ions/master_study/scans/MD2024/base_collider/xtrack_0000/../config.yaml .
mkdir xtrack_0000
cp -f /afs/cern.ch/work/c/cdroin/private/example_DA_study_runIII_ions/master_study/scans/MD2024/base_collider/xtrack_0000/config.yaml xtrack_0000
cd xtrack_0000
sed -i "s/..\/collider\/collider.json/\/afs\/cern.ch\/work\/c\/cdroin\/private\/example_DA_study_runIII_ions\/master_study\/scans\/MD2024\/base_collider\/xtrack_0000\/..\/collider\/collider.json/g" config.yaml
sed -i "s/..\/particles\/00.parquet/\/afs\/cern.ch\/work\/c\/cdroin\/private\/example_DA_study_runIII_ions\/master_study\/scans\/MD2024\/base_collider\/xtrack_0000\/..\/particles\/00.parquet/g" config.yaml
sed -i "s/tree_maker.log/\/afs\/cern.ch\/work\/c\/cdroin\/private\/example_DA_study_runIII_ions\/master_study\/scans\/MD2024\/base_collider\/xtrack_0000\/tree_maker.log/g" config.yaml
python /afs/cern.ch/work/c/cdroin/private/example_DA_study_runIII_ions/master_study/scans/MD2024/base_collider/xtrack_0000/2_configure_and_track.py > output_python.txt 2> error_python.txt
rm -f ../config.yaml
cp -f *.txt *.parquet *.yaml /afs/cern.ch/work/c/cdroin/private/example_DA_study_runIII_ions/master_study/scans/MD2024/base_collider/xtrack_0000
