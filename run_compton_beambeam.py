import os
import sys
import time
import numpy as np
import pandas as pd
import xtrack as xt
import xpart as xp
import xfields as xf
import xcoll as xc
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python run.py <jobID>")
    sys.exit(1)

jobID = int(sys.argv[1])

#################################################################
# Laser-interaction
#################################################################
CAIN_PATH = '/home/drebot/Xsuite/cain_compiled.gcc'

CUSTOM_ELEMENTS = {} 
try:
    from xcain import laser_interaction as xcain
    print('XCain found, LaserInteraction will be available as a user element')
    CUSTOM_ELEMENTS['LaserInteraction'] = xcain.LaserInteraction
except ImportError:
    pass

#################################################################
# Constants
#################################################################
NEMITT_X = 64.2506e-06 # m (corresponds to 0.72 nm geometric emittance)
NEMITT_Y = 0.12761e-06 # m (cooresponds to 1.43 pm geometric emittance)
SIGMA_Z  = 16.7E-3     # m

HALF_XING = 15E-3

BUNCH_INTENSITY = 2E11

R_BEAMPIPE = 0.030 # m
R_IP = 0.010 # m

IP_NAMES = ['ip', 'ip:1', 'ip:3', 'ip:5']

ELEMENT_START = 'qf9f_exit' # Exit of a QF in the Dx-free RF straight in PH

N_TURNS = 10
N_NOMINAL_MACROPARTICLES = 2000
WEAK_BEAM_ASYMMETRY = 0.10 # This means that the weak beam (the one that we track, e+) has 10% more bunch intensity
TARGET_REMOVED_FRACTION = 0.10 # This means that we turn off laser when 10% is removed

#################################################################
# File paths
#################################################################
home_fpath = os.getcwd() # Absolute path is recommended for HTCondor
env_fpath = 'fccee_z.json' # Absolute path is recommended for HTCondor
colldb_fpath = 'fccee_z_double_phase.colldb.yaml' # Absolute path is recommended for HTCondor

#################################################################
# Output directories
#################################################################
# Create output directory
output_dir = os.path.join(home_fpath, f'Output')
os.makedirs(output_dir, exist_ok=True)

# Create job-specific subdirectory
job_dir = os.path.join(output_dir, f"Job.{jobID}")
os.makedirs(job_dir, exist_ok=True)

# Create logs subdirectory inside job directory
logs_dir = os.path.join(job_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)

# Create Outputdata subdirectory inside job directory
outputdata_dir = os.path.join(job_dir, "Outputdata")
os.makedirs(outputdata_dir, exist_ok=True)

#################################################################
# Load lattice
#################################################################
env = xt.load(env_fpath)
line = env.lines['fccee_p_ring']
tab = line.get_table()

#################################################################
# Install collimators
#################################################################
colldb = xc.CollimatorDatabase.from_yaml(colldb_fpath)

aperture = xt.LimitEllipse(a=R_BEAMPIPE, b=R_BEAMPIPE)
colldb.install_geant4_collimators(line=line, apertures=aperture, verbose=True)

#################################################################
# Install emittance monitor
#################################################################
# NOTE: the monitor is installed at the beginning of the RF straight section in PH
s_monitor = 55901.0
mon = xc.EmittanceMonitor.install(line=line, name="emittance_monitor", at_s=s_monitor, stop_at_turn=N_TURNS)

#################################################################
# Install bounding apertures around emittance monitor
#################################################################
aper_name = 'emittance_monitor_aper'
env.new(aper_name, xt.LimitEllipse, a=R_BEAMPIPE, b=R_BEAMPIPE)
env.new(aper_name + '..0', aper_name, mode='replica')
env.new(aper_name + '..1', aper_name, mode='replica')

line.insert([
    env.place(aper_name + '..0', at='emittance_monitor@start'),
    env.place(aper_name + '..1', at='emittance_monitor@end')
])

#################################################################
# Assign optics to collimators
#################################################################
twiss = line.twiss()
line.collimators.assign_optics(twiss=twiss)

#################################################################
# Install beam-laser IP
#################################################################
elements = {**vars(xt.beam_elements.elements), **CUSTOM_ELEMENTS}

s_laser = tab.rows['ip_laser'].s[0]

laser_def = {
    'name': 'laserinho',
    'at_s': s_laser,
    'type': 'LaserInteraction',
    'parameters': {
        'name': 'my_laserinho',
        'cain_path': CAIN_PATH,
        'laser_parameter': {
            'pulseE': 3.00,      # laser pulse energy [J]
            'angle_deg': 0,      # in degree
            'sigLrx': 80,       # [µm]
            'sigLry': 80,       # [µm]
            'laserwl': 800,      # wavelength [nm]
            'sigt': 300,         # pulse length [ps]
            'shifting_laser_x': 0,
            'shifting_laser_y': 0,
            'shifting_laser_s': 0,
            'shifting_laser_t': 0,
            'NPH': 0,            # 0=linear; >=1 nonlinear
            'N_t_steps': 250,    # typical 250–300 (linear)
            'STOKES_1': 0,
            'STOKES_2': 0,
            'STOKES_3': 1
        },
        'photon_file': os.path.join(outputdata_dir, 'photons.hdf'),
        'seed': jobID*2 + 1 # Always odd, always different for different jobs
    }
}

laser_params = {}
for param, value in laser_def['parameters'].items():
    try:
        laser_params[param] = float(value)
    except:
        laser_params[param] = value
elem_name = laser_def['name']
elem_obj = elements[laser_def['type']](**laser_params)
s_position = laser_def['at_s']

line.insert_element(at_s=float(s_position), element=elem_obj, name=elem_name)

#################################################################
# Install beam-beam elements (weak-strong)
#################################################################
n_slices = 301
slicer = xf.TempSlicer(n_slices=n_slices, sigma_z=SIGMA_Z, mode="shatilov")

tab_sigmas = twiss.get_beam_covariance(nemitt_x=NEMITT_X, nemitt_y=NEMITT_Y)

beambeam_placements = []
for ip_name in IP_NAMES:
    sigma_x = tab_sigmas['sigma_x', ip_name]
    sigma_px = tab_sigmas['sigma_px', ip_name]
    sigma_y = tab_sigmas['sigma_y', ip_name]
    sigma_py = tab_sigmas['sigma_py', ip_name]

    beambeam = xf.BeamBeamBiGaussian3D(
                #_context=context,
                config_for_update = None,
                other_beam_q0=-1, # charge of the other beam (-1 for electrons)
                phi=HALF_XING, # half-crossing angle in radians
                alpha=0, # crossing plane (put to 0, ok for this use case)
                # decide between round or elliptical kick formula
                min_sigma_diff = 1e-28,
                # slice intensity [num. real particles] n_slices inferred from length of this
                slices_other_beam_num_particles = slicer.bin_weights * BUNCH_INTENSITY,
                # unboosted strong beam moments
                slices_other_beam_zeta_center = slicer.bin_centers,
                slices_other_beam_Sigma_11    = n_slices*[sigma_x**2], # Beam sizes for the other beam, assuming the same is approximation
                slices_other_beam_Sigma_22    = n_slices*[sigma_px**2],
                slices_other_beam_Sigma_33    = n_slices*[sigma_y**2],
                slices_other_beam_Sigma_44    = n_slices*[sigma_py**2],
                # only if BS on
                slices_other_beam_zeta_bin_width_star_beamstrahlung = slicer.bin_widths_beamstrahlung / np.cos(HALF_XING),  # boosted dz
                # has to be set
                slices_other_beam_Sigma_12    = n_slices*[0],
                slices_other_beam_Sigma_34    = n_slices*[0],
                compt_x_min                   = 1e-4,
        )
    beambeam.iscollective = True # This flag disables beam-beam in Twiss

    name = f'beambeam_{ip_name}'
    env.elements[name] = beambeam.copy()
    beambeam_placements.append(env.place(name, at=f'{ip_name}@start'))

line.insert(beambeam_placements)

#################################################################
# Install beam-beam elements bounding apertures
#################################################################
tab = line.get_table()
beambeam_names = tab.rows['beambeam.*'].name

placements = []
for nn in beambeam_names:
    aper_base = f'{nn}_aper'
    env.new(aper_base, xt.LimitEllipse, a=R_IP, b=R_IP)
    for ii_aper, pos in enumerate(('start', 'end')):
        aper_name = f'{aper_base}..{ii_aper}'
        env.new(aper_name, aper_base, mode='replica')
        placements.append(env.place(aper_name, at=f'{nn}@{pos}'))

line.insert(placements)

#################################################################
# Prepare matched Gaussian bunch at ELEMENT_START
#################################################################
# NOTE: here we consider the optics without beam-beam
#       to be discussed what to do in the future
x_norm, px_norm = xp.generate_2D_gaussian(N_NOMINAL_MACROPARTICLES)
y_norm, py_norm = xp.generate_2D_gaussian(N_NOMINAL_MACROPARTICLES)

# The longitudinal closed orbit needs to be manually supplied for now
element_index = line.element_names.index(ELEMENT_START)
zeta_co = twiss['zeta', ELEMENT_START]
delta_co = twiss['delta', ELEMENT_START] 

zeta, delta = xp.generate_longitudinal_coordinates(
    line=line,
    num_particles=N_NOMINAL_MACROPARTICLES,
    distribution='gaussian',
    sigma_z=SIGMA_Z
)

particles = line.build_particles(
    _capacity=4*N_NOMINAL_MACROPARTICLES,
    weight=BUNCH_INTENSITY * WEAK_BEAM_ASYMMETRY / N_NOMINAL_MACROPARTICLES,
    x_norm=x_norm, px_norm=px_norm,
    y_norm=y_norm, py_norm=py_norm,
    zeta=zeta + zeta_co,
    delta=delta + delta_co,
    nemitt_x=NEMITT_X,
    nemitt_y=NEMITT_Y,
    at_element=ELEMENT_START
    )

particles.start_tracking_at_element = -1

#################################################################
# Configure radiation mode for tracking
#################################################################
line.configure_radiation(model='quantum',
                         model_beamstrahlung='quantum',
                         model_bhabha=None)

#################################################################
# Start Geant4 engine
#################################################################
seed = jobID
xc.geant4.engine.seed = seed
xc.geant4.engine.start(line=line)

#################################################################
# Track!
#################################################################
line.scattering.enable()

t0 = time.time()
laser_on = True
for turn in range(N_TURNS):
    print(f'\nStart turn {turn}, Surviving particles: {particles._num_active_particles}')

    line.track(particles, ele_start=ELEMENT_START, ele_stop=ELEMENT_START, num_turns=1)

    primaries = particles.filter(particles.particle_id == particles.parent_particle_id)
    total_weight_primaries = primaries.weight[primaries.state > 0].sum()

    if total_weight_primaries <= BUNCH_INTENSITY and laser_on:
        print(f'\nReached target removed fraction at turn {turn}. Turning off laser.')
        # Turn laser off by replacing laser element with simple marker
        env.elements['laser_off'] = xt.Marker()
        line.replace('laserinho', 'laser_off')
        laser_on = False
        continue

    if particles._num_active_particles == 0:
        print(f'All particles lost by turn {turn}, terminating.')
        break

print(f'Tracking {N_TURNS} turns took {time.time() - t0} s.')

line.scattering.disable()

#############################################################
# Stop Geant4 engine
#############################################################
xc.geant4.engine.stop()

#############################################################
# Aperture loss interpolation
#############################################################
LossMap = xc.LossMap(line,
                     part=particles,
                     line_is_reversed=False,
                     interpolation=0.1,
                     weights=None,
                     weight_function=None)

#############################################################
# Save xcoll loss map
#############################################################
lossmap_fpath = os.path.join(outputdata_dir, 'lossmap.json')
LossMap.to_json(lossmap_fpath)

#############################################################
# Extract and save info from emittance monitor
#############################################################
monitor = line['emittance_monitor']

# Extract quantities
turns = np.array(monitor.turns)
gemitt_x = np.array(monitor.gemitt_x)
gemitt_y = np.array(monitor.gemitt_y)
# Other quantities can be extracted: https://github.com/xsuite/xcoll/blob/main/xcoll/beam_elements/monitor.py

# Store in dataframe and save
# df_emittance = pd.DataFrame({'turn': turns, 'gemitt_x': gemitt_x, 'gemitt_y': gemitt_y})
sigma_x  = np.sqrt(monitor.x_x_var)
sigma_y  = np.sqrt(monitor.y_y_var)
sigma_z  = np.sqrt(monitor.zeta_zeta_var)

sigma_px = np.sqrt(monitor.px_px_var)
sigma_py = np.sqrt(monitor.py_py_var)
sigma_pz = np.sqrt(monitor.pzeta_pzeta_var)

df_emittance = pd.DataFrame({
    'turn': turns,
    'gemitt_x': gemitt_x,
    'gemitt_y': gemitt_y,
    'sigma_x': sigma_x,
    'sigma_y': sigma_y,
    'sigma_z': sigma_z,
    'sigma_px': sigma_px,
    'sigma_py': sigma_py,
    'sigma_pz': sigma_pz
})
# I would recomment .to_parquet for less memory usage
df_emittance.to_csv(os.path.join(outputdata_dir, 'beam_stats.csv'))