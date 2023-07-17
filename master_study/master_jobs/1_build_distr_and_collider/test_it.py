import xtrack as xt
import numpy as np
import pandas as pd

collider = xt.Multiline.from_json('./collider/collider.json')

collider.build_trackers()



collider.vars['on_alice_normalized'] = 1
collider.vars['on_lhcb_normalized'] = 1

assert np.abs(collider.vars['on_alice_normalized']._value)==1
assert np.abs(collider.vars['on_lhcb_normalized']._value)==1


twiss_b1 = collider['lhcb1'].twiss()
twiss_b2 = collider['lhcb2'].twiss().reverse()

survey_b1 = collider['lhcb1'].survey()
survey_b2 = collider['lhcb2'].survey().reverse()
 
for my_ip in ['on_alice_normalized','on_lhcb_normalized']:
    print(f'*****************\nValues for {my_ip} (polarity):')
    print(collider.vars[my_ip]._value)
    print(f'*****************\n')

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


mydf = my_table[['x', 'y', 'px', 'py', 'betx', 'bety', 'alfx', 'alfy'],
            f'ip{my_ip}'].to_pandas()
mydf.name = 'B1'
mydf

