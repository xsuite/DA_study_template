import numpy as np
import xmask as xm

def check_madx_lattices(mad):
    assert mad.globals["qxb1"] == mad.globals["qxb2"]
    assert mad.globals["qyb1"] == mad.globals["qyb2"]
    assert mad.globals["qpxb1"] == mad.globals["qpxb2"]
    assert mad.globals["qpyb1"] == mad.globals["qpyb2"]

    assert np.isclose(mad.table.summ.q1, mad.globals["qxb1"], atol=1e-02)
    assert np.isclose(mad.table.summ.q2, mad.globals["qyb1"], atol=1e-02)
    assert np.isclose(mad.table.summ.dq1, mad.globals["qpxb1"], atol=5e-01)
    assert np.isclose(mad.table.summ.dq2, mad.globals["qpyb1"], atol=5e-01)

    df = mad.table.twiss.dframe()
    for my_ip in [1, 2, 5, 8]:
        #assert np.isclose(df.loc[f"ip{my_ip}"].betx, mad.globals[f"betx_IP{my_ip}"], rtol=1e-02)
        #assert np.isclose(df.loc[f"ip{my_ip}"].bety, mad.globals[f"bety_IP{my_ip}"], rtol=1e-02)
        assert np.isclose(df.loc[f"ip{my_ip}"].betx, mad.globals[f"betxIP{my_ip}b1"], rtol=1e-02)
        assert np.isclose(df.loc[f"ip{my_ip}"].bety, mad.globals[f"betyIP{my_ip}b1"], rtol=1e-02)

    mad.input('exec, crossing_save;')
    mad.input('exec, crossing_disable;')
    mad.twiss()
    df = mad.table.twiss.dframe()

    assert df["x"].std() < 1e-6
    assert df["y"].std() < 1e-6
    mad.input('exec, crossing_restore;')


def check_xsuite_lattices(my_line):
    tw = my_line.twiss(method="6d", matrix_stability_tol=100)
    print(f"--- Now displaying Twiss result at all IPS for line {my_line}---")
    print(tw[:, "ip.*"])
    # print qx and qy
    print(f"--- Now displaying Qx and Qy for line {my_line}---")
    print(tw.qx, tw.qy)

def build_sequence(mad, mylhcbeam, beam_config, ignore_cycling=False, slice_factor = 8):
    # Select beam
    mad.input(f"mylhcbeam = {mylhcbeam}")
    mad.input('option, -echo,warn, -info;')

    # optics dependent macros (for splitting)
    mad.call('acc-models-lhc/runII/2018/toolkit/macro.madx')

    # # Redefine macros
    _redefine_crossing_save_disable_restore(mad)

    # # optics independent macros
    # mad.call('tools/optics_indep_macros.madx')

    assert mylhcbeam in [1, 2, 4], "Invalid mylhcbeam (it should be in [1, 2, 4])"

    if mylhcbeam in [1, 2]:
        mad.call('acc-models-lhc/runII/2018/lhc_as-built.seq')
    else:
        mad.call('acc-models-lhc/runII/2018/lhcb4_as-built.seq')

    # New IR7 MQW layout and cabling
    mad.call('acc-models-lhc/runIII/RunIII_dev/IR7-Run3seqedit.madx')

    # Makethin part
    if slice_factor > 0:
        # the variable in the macro is slice_factor
        mad.input(f'slicefactor={slice_factor};')
        mad.call('acc-models-lhc/runII/2018/toolkit/myslice.madx')
        for my_sequence in list(mad.sequence):
          xm.attach_beam_to_sequence(mad.sequence[my_sequence],int(my_sequence[-1]), beam_config[my_sequence])
        #mad.beam()
        for my_sequence in ['lhcb1','lhcb2']:
            if my_sequence in list(mad.sequence):
                mad.input(f'use, sequence={my_sequence}; makethin,'
                     f'sequence={my_sequence}, style=teapot, makedipedge=true;')
    else:
        print('WARNING: The sequences are not thin!')

    # Cycling w.r.t. to IP3 (mandatory to find closed orbit in collision in the presence of errors)
    if not ignore_cycling:
        for my_sequence in ['lhcb1','lhcb2']:
            if my_sequence in list(mad.sequence):
                mad.input(f'seqedit, sequence={my_sequence}; flatten;'
                            'cycle, start=IP3; flatten; endedit;')


def apply_optics(mad, optics_file):
    mad.call(optics_file)
    mad.call('ir7_strengths.madx')
    mad.input('on_alice := on_alice_normalized * 7000. * 82. /nrj;')
    mad.input('on_lhcb := on_lhcb_normalized * 7000. * 82. /nrj;')

def _redefine_crossing_save_disable_restore(mad):

    mad.input('''
    crossing_save: macro = {
    on_x1_aux=on_x1;on_sep1_aux=on_sep1;on_a1_aux=on_a1;on_o1_aux=on_o1;
    on_x2_aux=on_x2;on_sep2_aux=on_sep2;on_a2_aux=on_a2;on_o2_aux=on_o2; on_oe2_aux=on_oe2;
    on_x5_aux=on_x5;on_sep5_aux=on_sep5;on_a5_aux=on_a5;on_o5_aux=on_o5;
    on_x8_aux=on_x8;on_sep8_aux=on_sep8;on_a8_aux=on_a8;on_o8_aux=on_o8;
    on_disp_aux=on_disp;
    on_alice_aux=on_alice;
    on_lhcb_aux=on_lhcb;
    on_ov2_aux=on_ov2;
    on_ov5_aux=on_ov5;
    };

    crossing_disable: macro={
    on_x1=0;on_sep1=0;on_a1=0;on_o1=0;
    on_x2=0;on_sep2=0;on_a2=0;on_o2=0;on_oe2=0;
    on_x5=0;on_sep5=0;on_a5=0;on_o5=0;
    on_x8=0;on_sep8=0;on_a8=0;on_o8=0;
    on_disp=0;
    on_alice=0; on_lhcb=0;
    on_ov2=0;on_ov5=0;
    };

    crossing_restore: macro={
    on_x1=on_x1_aux;on_sep1=on_sep1_aux;on_a1=on_a1_aux;on_o1=on_o1_aux;
    on_x2=on_x2_aux;on_sep2=on_sep2_aux;on_a2=on_a2_aux;on_o2=on_o2_aux; on_oe2=on_oe2_aux;
    on_x5=on_x5_aux;on_sep5=on_sep5_aux;on_a5=on_a5_aux;on_o5=on_o5_aux;
    on_x8=on_x8_aux;on_sep8=on_sep8_aux;on_a8=on_a8_aux;on_o8=on_o8_aux;
    on_disp=on_disp_aux;
    on_alice=on_alice_aux; on_lhcb=on_lhcb_aux;
    on_ov2=on_ov2_aux;
    on_ov5=on_ov5_aux;
    };
    ''')

