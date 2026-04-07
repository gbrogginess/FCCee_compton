import xtrack as xt

#################################################################
# File paths
#################################################################
home_fpath = '/home/gbroggi/forIllya_new/' # Change according to your setup
env_fpath = 'fccee_z_plain.json'
colldb_fpath = 'fccee_z_double_phase.colldb.yaml'

#################################################################
# Load lattice
#################################################################
env = xt.load(env_fpath)
line = env.lines['fccee_p_ring']
tab = line.get_table()

#################################################################
# Insert a marker for the laser IP
#################################################################
laser_dipole_slice_name = 'dl1a:1901..1'

s_laser_ip = tab.rows[laser_dipole_slice_name].s_center[0]

env.new('ip_laser', xt.Marker)
line.insert([
    env.place('ip_laser', at=s_laser_ip)
])

#################################################################
# Install bounding apertures around dipole slices
#################################################################
laser_dipole_name = laser_dipole_slice_name.split('..')[0]

placements = []
for ii_slice in (0, 1):
    slice_name = f'{laser_dipole_slice_name}..{ii_slice}'
    for ii_aper, pos in enumerate(('start', 'end')):
        aper_name = f'{slice_name}_aper..{ii_aper}'
        env.new(aper_name, laser_dipole_name + '_aper', mode='replica')
        placements.append(env.place(aper_name, at=f'{slice_name}@{pos}'))

line.insert(placements)

#################################################################
# Taper lattice again after slicing
#################################################################
line.build_tracker()
line['changeq_on']=0
line['emity_on']=0
line.compensate_radiation_energy_loss()
line['changeq_on']=1.0
line['emity_on']=1.0

#################################################################
# Save
#################################################################
env.to_json(home_fpath + 'fccee_z.json')