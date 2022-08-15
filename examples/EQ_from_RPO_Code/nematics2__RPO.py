import h5py
import scipy.io
from scipy import sparse
from scipy import optimize

import numpy as np
from mpi4py import MPI
from dedalus import public as de

import os
import shutil
import glob
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
import timeit
start = timeit.default_timer()

from dedalus.tools.config import config
config['logging']['stdout_level'] = 'none'

#---------------------------------------- #
# Input file: initial guess for Newton solver
#---------------------------------------- #
input_filename = './solution_init.h5'
input_file = h5py.File(input_filename, 'r')
#---------------------------------------- #

#---------------------------------------- #
# Log file
#---------------------------------------- #
if os.path.exists('./solver.out'):
	os.remove('./solver.out')
log_file = open('solver.out', 'w')
#---------------------------------------- #

#---------------------------------------- #
# Solver parameters
#---------------------------------------- #
ECS_id = '' # (optional) identifier for ECS
d_input_mode = 'c' # Data input format; 'c' = coefficient representation and 'g' = grid representation
output_full_trajectory = 0 # After the Newton solver converges, the code automatically time integrates over a single period and outputs
                                       # time series of some channel averages. There is also the option to output the full field data by setting 'output_full_trajectory' to 1

Ra = 1.0*np.array(input_file.get('/params/Ra'))+ 0.005 #continuation from known solution
time_limit = 4
tolerance = 1e-10 # Convergence criterion
newton_iterations = 100
krylov_dim = 20 # Max Krylov subspace dimension
trust_radius = 0.1
dt_nominal = 0.01 # Nominal timestep. Actual timestep will differ due to rounding, so as to ensure an integer number of timesteps in the evaluation of the flow map phi
timestepper = de.timesteppers.RK222

# Resolution of input data. Note that for real-valued domains, Dedalus only stores half the number of Fourier modes; 
# with the remaining half determined by complex conjugation.
NX_input = int(1.0*np.array(input_file.get('/params/NX'))) # Fourier modes
NY_input = int(1.0*np.array(input_file.get('/params/NY'))) # Chebyshev modes
NXH_input = int(NX_input/2)

height = 1.0*np.array(input_file.get('/params/height')) # Channel height
width = 1.0*np.array(input_file.get('/params/width')) # Channel width
NX = NX_input # Change these if needed
NY = NY_input #
NXH = int(NX/2)
#---------------------------------------- #

#---------------------------------------- #
# Bases and domain
#---------------------------------------- #
dealias_fac = 2 # Dealiasing 
x_basis = de.Fourier('x', NX, interval=(-0.5*width, 0.5*width), dealias = dealias_fac)
y_basis = de.Chebyshev('y', NY, interval=(0, height), dealias = dealias_fac)
domain = de.Domain([x_basis, y_basis], grid_dtype = np.float64)
xg = domain.grid(0)
yg = domain.grid(1)
#---------------------------------------- #

#----------------------------------------------------------------------------------------- #
#----------------------------------------------------------------------------------------- #
# SOLVER MODES #
#----------------------------------------------------------------------------------------- #
#----------------------------------------------------------------------------------------- #

# --------> RPO MODE (DEFAULT) <---------- #
# T_guess = np.array(input_file.get('/T')) # Initial guess for the period 'T'
# d_guess = np.array(input_file.get('/d')) # Initial guess for the shift 'd'
#---------------------------------------- #

# --------> PO (PERIODIC ORBIT) MODE <---------- #
# For POs, set the shift to something close to 0. Strictly speaking, it should be exactly 0, but
# the current code requires a finite value to avoid division by 0.
#T_guess = np.array(input_file.get('/T')) # Initial guess for the period 'T'
#d_guess = 1e-12
#---------------------------------------- #

# --------> TW (TRAVELING WAVE) MODE <---------- #
# For TWs, there is 1 unknown parameter in addition to the field data: namely,
# the wave speed 'c'. To calculate TWs in this script, the initial value for 'T' is
# taken to be 1.0 -- about 100 timesteps -- and 'd' is initialized to 'c*T'.
# The value for 'T' is meant to be "small but not too small". If it is too large, then
# the solver may leave the vicinity of the TW during the time integration (for unstable TWs). 
# If it is too small, then the Newton solver will seek the trivial fixed point T = 0.
#T_guess = 1.0
#c_guess = c_guess # Input initial guess for the wave speed here
#d_guess = c_guess*T_guess
#---------------------------------------- #

# --------> EQ (EQUILIBRIA) MODE <---------- #
# Equilibria are treated similarly to traveling waves, except that 'd' is initialized as for a PO,
# that is, approximately zero within working precision.)
T_guess = 1.0
d_guess = 1e-12
#---------------------------------------- #

#----------------------------------------------------------------------------------------- #
#----------------------------------------------------------------------------------------- #

#---------------------------------------- #
# Will be used to rescale 'T' and 'd' 
# when executing the Newton solver
#---------------------------------------- #
tfac = 3*T_guess
dfac = 5*d_guess
#---------------------------------------- #

#---------------------------------------- #
# Input field data for the initial guess
#---------------------------------------- #
if d_input_mode == 'g' and NX_input == NX and NY_input == NY:
	dataQAA_init = np.array(input_file.get('/QAA')) # (x,x) component of Q-tensor in grid space
	dataQAB_init = np.array(input_file.get('/QAB')) # (x,y) component of Q-tensor in grid space
	dataU_init = np.array(input_file.get('/U')) # x-velocity in grid space
	dataV_init = np.array(input_file.get('/V')) # y-velocity in grid space

	QAA_field = domain.new_field()
	QAB_field = domain.new_field()
	U_field = domain.new_field()
	V_field = domain.new_field()

	QAA_field['g'] = dataQAA_init
	QAB_field['g'] = dataQAB_init
	U_field['g'] = dataU_init
	V_field['g'] = dataV_init

	QAA_field.require_coeff_space()
	QAB_field.require_coeff_space()
	U_field.require_coeff_space()
	V_field.require_coeff_space()
	
	QAA_data_init_coeff = np.copy(QAA_field.data)
	QAB_data_init_coeff = np.copy(QAB_field.data)
	U_data_init_coeff = np.copy(U_field.data)
	V_data_init_coeff = np.copy(V_field.data)
elif d_input_mode == 'c':
	dataQAA_init = np.array(input_file.get('/QAA_coeff')) # (x,x) component of Q-tensor in coefficient space
	dataQAB_init = np.array(input_file.get('/QAB_coeff')) # (x,y) component of Q-tensor in coefficient space
	dataU_init = np.array(input_file.get('/U_coeff')) # x-velocity in coefficient space
	dataV_init = np.array(input_file.get('/V_coeff')) # y-velocity in coefficient space 

	QAA_data_init_coeff = np.zeros((NXH, NY), dtype = np.complex128)
	QAB_data_init_coeff = np.zeros((NXH, NY), dtype = np.complex128)
	U_data_init_coeff = np.zeros((NXH, NY), dtype = np.complex128)
	V_data_init_coeff = np.zeros((NXH, NY), dtype = np.complex128)

	# Use 'min(NXH,NXH_input)' to cover both cases when NXH_input > NXH or NXH_input < NXH
	QAA_data_init_coeff[0:min(NXH,NXH_input),0:min(NY,NY_input)] = np.copy(dataQAA_init)[0:min(NXH,NXH_input),0:min(NY,NY_input)]
	QAB_data_init_coeff[0:min(NXH,NXH_input),0:min(NY,NY_input)] = np.copy(dataQAB_init)[0:min(NXH,NXH_input),0:min(NY,NY_input)]
	U_data_init_coeff[0:min(NXH,NXH_input),0:min(NY,NY_input)] = np.copy(dataU_init)[0:min(NXH,NXH_input),0:min(NY,NY_input)]
	V_data_init_coeff[0:min(NXH,NXH_input),0:min(NY,NY_input)] = np.copy(dataV_init)[0:min(NXH,NXH_input),0:min(NY,NY_input)]
else:
	if d_input_mode == 'g' and (not NX_input == NX or not NY_input == NY):
		print('Error: Changing resolution on input data in grid space is not currently supported.')
	else:
		print('Error: Data input mode not recognized.')
#---------------------------------------- #

#---------------------------------------- #
# Other parameters 
#---------------------------------------- #
n_fields = 4 # Number of physical fields (here: QAA, QAB, U, V)
MY = n_fields*NX*NY # Total number of degrees of freedom
d_tol = 1e-7 # Step size for finite-difference in computing GMRES matrix-vector product
delta = 1e-7 # Step size for finite-difference approximation of the time derivative of the equations of motion
d_tol_translation = 1e-7 # Step size for finite-difference approximation of the infinitesimal generator of translations
n_timesteps = 8 # Number of timesteps used for the finite difference estimate  of the time derivative of the equations of motion

Re = 0.0136
Er = 1.0
#---------------------------------------- #

def VectorToGrid(array_in):
	array_out = np.zeros((NXH, NY), dtype = np.complex128)
	for i in range(NXH):
		ih = NXH+i
		array_out[i,:] = array_in[(i*NY):(i+1)*NY] + 1j*array_in[(ih*NY):(ih+1)*NY]
	return array_out
	
def GridToVector(array_in):
	array_out = np.zeros(NX*NY, dtype = np.float64)
	for i in range(NXH):
		ih = NXH+i
		array_out[(i*NY):(i+1)*NY] = array_in[i,:].real
		array_out[(ih*NY):(ih+1)*NY] = array_in[i,:].imag
	return array_out

# 'f' is the state vector. Here we assign it the appropriate initial data. Note that we have to convert the Dedalus grid representation
# into a real-valued vector
f = np.zeros(MY+2)
qaa_begin = 0
qaa_end = qaa_begin + (NX*NY)
qab_begin = qaa_end
qab_end = qab_begin + (NX*NY)
u_begin = qab_end
u_end = u_begin + (NX*NY)
v_begin = u_end
v_end = v_begin + (NX*NY)
f[0] = T_guess/tfac  # 'f[0]' gives the normalized period, i.e. the current value of 'T' divided by the initial guess, 'T_guess'
f[1] = d_guess/dfac # similarly for the shift 'd'
f[(qaa_begin+2):(qaa_end+2)] = GridToVector(np.copy(QAA_data_init_coeff))
f[(qab_begin+2):(qab_end+2)] = GridToVector(np.copy(QAB_data_init_coeff))
f[(u_begin+2):(u_end+2)] = GridToVector(np.copy(U_data_init_coeff))
f[(v_begin+2):(v_end+2)] = GridToVector(np.copy(V_data_init_coeff))

# Initial iterate for GMRES; use later.
z0 = np.zeros(MY+2)
zn = np.copy(z0)
	
print('\n')
print('---------------------------------')
print('--- Parameters ------------------')
print('---------------------------------')
print('Ra = ' + str(Ra))
print('Er = ' + str(Er))
print('Re = ' + str(Re))
print('NX = ' + str(NX))
print('NY = ' + str(NY))
print('height = ' + str(height))
print('width = ' + str(width))
print('T0 = ' + str(T_guess))
print('d0 = ' + str(d_guess))
print('timestep = ' + str(dt_nominal))
print('trust radius = ' + str(trust_radius))
print('---------------------------------')

log_file.writelines('---------------------------------' + '\n')
log_file.writelines('--- Parameters ------------------' + '\n')
log_file.writelines('---------------------------------' + '\n')
log_file.writelines('Ra = ' + str(Ra) + '\n')
log_file.writelines('Er = ' + str(Er) + '\n')
log_file.writelines('Re = ' + str(Re) + '\n')
log_file.writelines('NX = ' + str(NX) + '\n')
log_file.writelines('NY = ' + str(NY) + '\n')
log_file.writelines('height = ' + str(height) + '\n')
log_file.writelines('width = ' + str(width) + '\n')
log_file.writelines('T0 = ' + str(T_guess) + '\n')
log_file.writelines('d0 = ' + str(d_guess) + '\n')
log_file.writelines('timestep = ' + str(dt_nominal) + '\n')
log_file.writelines('trust radius = ' + str(trust_radius) + '\n')
log_file.writelines('---------------------------------' + '\n')
log_file.writelines('\n')

# Shift field data by 'd' units and convert to vector format
def TransformG(array_in, d):
	data_temp_c = np.copy(array_in)
	for i in range(NXH):
		data_temp_c[i,:] = np.exp(2*1j*i*np.pi*(d/width))*data_temp_c[i,:]
	return GridToVector(data_temp_c)

# Compute infinitesimal generator of x-translation
def dTransform(array_in):
	array_temp = np.copy(array_in)
	
	QAA_data_temp = VectorToGrid(array_temp[qaa_begin:qaa_end])
	QAB_data_temp = VectorToGrid(array_temp[qab_begin:qab_end])
	U_data_temp = VectorToGrid(array_temp[u_begin:u_end])
	V_data_temp = VectorToGrid(array_temp[v_begin:v_end])
	
	array_out = np.zeros(MY)
	array_out[qaa_begin:qaa_end] = (TransformG(QAA_data_temp, d_tol_translation) - array_temp[qaa_begin:qaa_end])/d_tol_translation
	array_out[qab_begin:qab_end] = (TransformG(QAB_data_temp, d_tol_translation) - array_temp[qab_begin:qab_end])/d_tol_translation
	array_out[u_begin:u_end] = (TransformG(U_data_temp, d_tol_translation) - array_temp[u_begin:u_end])/d_tol_translation
	array_out[v_begin:v_end] = (TransformG(V_data_temp, d_tol_translation) - array_temp[v_begin:v_end])/d_tol_translation
	return array_out




''' Function phi:

	Description
	---------------------
	Flow map phi(array_in, T, d)
	---------------------

	Parameters
	---------------------
	array_in :
		variable type: Real-valued, 1d numpy data array with dimension MY = n_fields*NX*NY
		description: state vector x0 in coefficient format
	T :
		variable type: float
		description: time interval over which to apply the flow map
	d :
		variable type: float
		description: shift to apply at the end of the time integration
	---------------------
'''
def phi(array_in, T, d):
	x_basis_phi = de.Fourier('x', NX, interval=(-0.5*width, 0.5*width), dealias = dealias_fac)
	y_basis_phi = de.Chebyshev('y', NY, interval=(0, height), dealias = dealias_fac)
	domain_phi = de.Domain([x_basis_phi, y_basis_phi], grid_dtype = np.float64)
	problem_phi = de.IVP(domain_phi, variables=['p','U','V','Ux','Uy','Vx','Vy','QAA','QAAx','QAAy','QAB','QABx','QABy'])
	problem_phi.meta[:]['y']['dirichlet'] = True
	problem_phi.parameters['width'] = width
	problem_phi.parameters['height'] = height
	problem_phi.parameters['RaEr'] = Ra/Er
	problem_phi.parameters['Re'] = Re
	problem_phi.add_equation("Ux - dx(U) = 0")
	problem_phi.add_equation("Uy - dy(U) = 0")
	problem_phi.add_equation("Vx - dx(V) = 0")
	problem_phi.add_equation("Vy - dy(V) = 0")
	problem_phi.add_equation("QAAx - dx(QAA) = 0")
	problem_phi.add_equation("QAAy - dy(QAA) = 0")
	problem_phi.add_equation("QABx - dx(QAB) = 0")
	problem_phi.add_equation("QABy - dy(QAB) = 0")
	problem_phi.add_equation("dt(QAA)-dx(QAAx)-dy(QAAy)-QAA=-10*QAA*QAA*QAA-10*QAA*QAB*QAB-QAAx*U-QAAy*V+QAB*Uy-QAB*Vx")
	problem_phi.add_equation("dt(QAB)-dx(QABx)-dy(QABy)-QAB=-10*QAA*QAA*QAB-10*QAB*QAB*QAB-QAA*Uy+QAA*Vx-QABx*U-QABy*V")
	problem_phi.add_equation("Re*dt(U)+dx(p)-2*dx(Ux)-dy(Uy)-dx(Vy)+RaEr*QAAx+RaEr*QABy=-Re*U*Ux-Re*Uy*V")
	problem_phi.add_equation("Re*dt(V)+dy(p)-2*dy(Vy)-dx(Uy)-dx(Vx)-RaEr*QAAy+RaEr*QABx=-Re*U*Vx-Re*V*Vy")
	problem_phi.add_equation("Ux + Vy = 0")
	problem_phi.add_bc("left(U) = 0")
	problem_phi.add_bc("left(V) = 0")
	problem_phi.add_bc("left(QAA) = -0.5")
	problem_phi.add_bc("left(QAB) = 0")
	problem_phi.add_bc("right(U) = 0")
	problem_phi.add_bc("right(V) = 0", condition="(nx != 0)")
	problem_phi.add_bc("left(p) = 0", condition="(nx == 0)")
	problem_phi.add_bc("right(QAA) = -0.5")
	problem_phi.add_bc("right(QAB) = 0")
	solver_phi = problem_phi.build_solver(timestepper)

	QAA = solver_phi.state['QAA']
	QAB = solver_phi.state['QAB']
	U = solver_phi.state['U']
	V = solver_phi.state['V']
	
	QAA['c'] = VectorToGrid(array_in[qaa_begin:qaa_end])
	QAB['c'] = VectorToGrid(array_in[qab_begin:qab_end])
	U['c'] = VectorToGrid(array_in[u_begin:u_end])
	V['c'] = VectorToGrid(array_in[v_begin:v_end])
	
	num_steps = int(T/dt_nominal)
	dt = T/num_steps
	for i in range(num_steps):
		solver_phi.step(dt)

	array_out = np.zeros(MY)
	array_out[qaa_begin:qaa_end] = TransformG(np.copy(QAA.data), d)
	array_out[qab_begin:qab_end] = TransformG(np.copy(QAB.data), d)
	array_out[u_begin:u_end] = TransformG(np.copy(U.data), d)
	array_out[v_begin:v_end] = TransformG(np.copy(V.data), d)
	return array_out
''' '''



''' Function Dphi_prod:

	Description
	---------------------
	Matrix-vector product, math notation: (d phi / dx_0) * delta_x 
	---------------------

	Parameters
	---------------------
	array_base :
		variable type: Real-valued, 1d numpy data array with dimension MY = n_fields*NX*NY
		description: initial (base) vector x0 in coefficient format
	array_pert :
		variable type: Real-valued, 1d numpy data array with dimension MY = n_fields*NX*NY
		description: 'delta_x' in the equation for the Matrix-vector product
	T :
		variable type: float
		description: time interval over which to apply the flow map
	d :
		variable type: float
		description: shift to apply at the end of the time integration
	---------------------
'''
def Dphi_prod(array_base, array_pert, T, d):
	if np.linalg.norm(array_pert) == 0:
		epsilon = 1
		return np.zeros(MY)
	else:
		epsilon = d_tol*np.linalg.norm(array_base)/np.linalg.norm(array_pert)
	array_init = array_base + epsilon*array_pert
	array_final = phi(array_init, T, d)
	array_final_base = np.copy(phi_base)
	array_out = (array_final-array_final_base)/epsilon
	
	return np.copy(array_out)
''' '''



''' Function applyLinearOperator:

	Description
	---------------------
	Linearized operator for the Newton iteration
	---------------------

	Parameters
	---------------------
	array_base :
		variable type: Real-valued, 1d numpy data array with dimension MY + 2 = n_fields*NX*NY + 2
		description: base vector about which the linearization is performed, including the period 'T' and shift 'd'
	array_pert :
		variable type: Real-valued, 1d numpy data array with dimension MY + 2 = n_fields*NX*NY + 2
		description: linear perturbation we are solving for, i.e.,  'dx' in the equation 'A*dx = b'
	---------------------
'''
def applyLinearOperator(array_base, array_pert):
	T0 = array_base[0]*tfac
	delta_T = array_pert[0]*tfac
	d0 = array_base[1]*dfac
	delta_d = array_pert[1]*dfac
	
	x_base = np.copy(array_base[2:])
	delta_x = np.copy(array_pert[2:])
	
	array_out = np.zeros(MY+2)
	array_out[0] = np.matmul(np.conj(RHS(x_base)), delta_x)
	array_out[1] = np.matmul(np.conj(dTransform(x_base)), delta_x)
	array_out[2:] = Dphi_prod(x_base, delta_x, T0, d0) - delta_x + RHS(np.copy(phi_base))*delta_T + dTransform(phi_base)*delta_d
	return array_out
''' '''



''' Function RHS:

	Description
	---------------------
	Calculate time-derivative in the final state via finite difference (equal to the r.h.s. of the equations of motion)
	---------------------

	Parameters
	---------------------
	array_in :
		variable type: Real-valued, 1d numpy data array with dimension MY = n_fields*NX*NY
		description: input vector
	---------------------
'''
def RHS(array_in):
	x_basis_phi = de.Fourier('x', NX, interval = (-0.5*width, 0.5*width), dealias = dealias_fac)
	y_basis_phi = de.Chebyshev('y', NY, interval = (0, height), dealias = dealias_fac)
	domain_phi = de.Domain([x_basis_phi, y_basis_phi], grid_dtype = np.float64)
	problem_phi = de.IVP(domain_phi, variables=['p','U','V','Ux','Uy','Vx','Vy','QAA','QAAx','QAAy','QAB','QABx','QABy'])
	problem_phi.meta[:]['y']['dirichlet'] = True
	problem_phi.parameters['width'] = width
	problem_phi.parameters['height'] = height
	problem_phi.parameters['RaEr'] = Ra/Er
	problem_phi.parameters['Re'] = Re
	problem_phi.add_equation("Ux - dx(U) = 0")
	problem_phi.add_equation("Uy - dy(U) = 0")
	problem_phi.add_equation("Vx - dx(V) = 0")
	problem_phi.add_equation("Vy - dy(V) = 0")
	problem_phi.add_equation("QAAx - dx(QAA) = 0")
	problem_phi.add_equation("QAAy - dy(QAA) = 0")
	problem_phi.add_equation("QABx - dx(QAB) = 0")
	problem_phi.add_equation("QABy - dy(QAB) = 0")
	problem_phi.add_equation("dt(QAA)-dx(QAAx)-dy(QAAy)-QAA=-10*QAA*QAA*QAA-10*QAA*QAB*QAB-QAAx*U-QAAy*V+QAB*Uy-QAB*Vx")
	problem_phi.add_equation("dt(QAB)-dx(QABx)-dy(QABy)-QAB=-10*QAA*QAA*QAB-10*QAB*QAB*QAB-QAA*Uy+QAA*Vx-QABx*U-QABy*V")
	problem_phi.add_equation("Re*dt(U)+dx(p)-2*dx(Ux)-dy(Uy)-dx(Vy)+RaEr*QAAx+RaEr*QABy=-Re*U*Ux-Re*Uy*V")
	problem_phi.add_equation("Re*dt(V)+dy(p)-2*dy(Vy)-dx(Uy)-dx(Vx)-RaEr*QAAy+RaEr*QABx=-Re*U*Vx-Re*V*Vy")
	problem_phi.add_equation("Ux + Vy = 0")
	problem_phi.add_bc("left(U) = 0")
	problem_phi.add_bc("left(V) = 0")
	problem_phi.add_bc("left(QAA) = -0.5")
	problem_phi.add_bc("left(QAB) = 0")
	problem_phi.add_bc("right(U) = 0")
	problem_phi.add_bc("right(V) = 0", condition="(nx != 0)")
	problem_phi.add_bc("left(p) = 0", condition="(nx == 0)")
	problem_phi.add_bc("right(QAA) = -0.5")
	problem_phi.add_bc("right(QAB) = 0")
	solver_phi = problem_phi.build_solver(de.timesteppers.SBDF1)
	
	QAA = solver_phi.state['QAA']
	QAB = solver_phi.state['QAB']
	U = solver_phi.state['U']
	V = solver_phi.state['V']
	
	QAA['c'] = VectorToGrid(array_in[qaa_begin:qaa_end])
	QAB['c'] = VectorToGrid(array_in[qab_begin:qab_end])
	U['c'] = VectorToGrid(array_in[u_begin:u_end])
	V['c'] = VectorToGrid(array_in[v_begin:v_end])
	
	QAA_data_initial = np.copy(QAA.data)
	QAB_data_initial = np.copy(QAB.data)
	U_data_initial = np.copy(U.data)
	V_data_initial = np.copy(V.data)
	
	for i in range(n_timesteps):
		solver_phi.step(delta)

	array_out = np.zeros(MY)
	array_out[qaa_begin:qaa_end] = GridToVector((np.copy(QAA.data) - QAA_data_initial)/(n_timesteps*delta))
	array_out[qab_begin:qab_end] = GridToVector((np.copy(QAB.data) - QAB_data_initial)/(n_timesteps*delta))
	array_out[u_begin:u_end] = GridToVector((np.copy(U.data) - U_data_initial)/(n_timesteps*delta))
	array_out[v_begin:v_end] = GridToVector((np.copy(V.data) - V_data_initial)/(n_timesteps*delta))
	return array_out
''' '''




''' Function applyNonLinearOperator:

	Description
	---------------------
	Full nonlinear operator
	---------------------

	Parameters
	---------------------
	array_in :
		variable type: Real-valued, 1d numpy data array with dimension MY + 2 = n_fields*NX*NY + 2
		description: input vector
	---------------------
'''
def applyNonLinearOperator(array_in):
	array_out = np.zeros(MY+2)
	array_out[0] = 0
	array_out[1] = 0
	array_out[2:] = -phi(array_in[2:], array_in[0]*tfac, array_in[1]*dfac) + array_in[2:]
	return array_out
''' '''


def arnoldi_iteration(x_base, T, d, r, n:int):
	Q = np.zeros((r.size, n+1))
	H = np.zeros((n+1, n))
	Q[:,0] = r/np.linalg.norm(r)
	for k in range(1, n + 1):
		Q[:,k] = Dphi_prod(x_base, Q[:, k - 1], T, d)
		for j in range(0, k):
			H[j, k-1] = np.matmul(np.conj(Q[:,j]), Q[:,k])
			Q[:,k] = Q[:,k] - H[j, k-1]*Q[:,j]
		H[k, k-1] = np.linalg.norm(Q[:,k])
		Q[:,k] = Q[:,k]/H[k, k-1]
		
	return Q, H
	
def arnoldi_iteration_inner(x_base, Q, k:int):
	Qk = applyLinearOperator(x_base, Q[:, k - 1])
	Hk = np.zeros(k+1)
	for j in range(0, k):
		Hk[j] = np.matmul(np.conj(Q[:,j]), Qk)
		Qk = Qk- Hk[j]*Q[:,j]
	Hk[k] = np.linalg.norm(Qk)
	Qk = Qk/Hk[k]

	return Qk, Hk
	
def Hookstep(H_, beta_, k_, tr):
	e1 = np.zeros(k_+1)
	e1[0] = beta_
	
	def fun(x_, F):
		r = np.matmul(F, x_) + e1
		return np.matmul(r, r)

	def Jacobian(x_, F):
		return 2*np.matmul(np.matmul(np.transpose(F), F), x_) + 2*np.matmul(np.transpose(F), e1)
	
	def constraint(x_):
		return tr*tr - np.matmul(np.transpose(x_), x_)
	
	def constraintJac(x_):
		return -2*x_
	
	ineq_cons = {'type': 'ineq','fun' : constraint,'jac' : constraintJac}
	
	w_init = np.zeros(k_)
	w_init[0] = 1e-3
	
	res = scipy.optimize.minimize(fun, w_init, args=(H_[0:k_+1,0:k_]), method='SLSQP', jac = Jacobian,
		constraints=(ineq_cons), options={'ftol': 1e-34, 'disp': False, 'maxiter': 100000000}, bounds=None)
	return res

def GMRES(x_base, x0, b, kmax, tr):
	xk = np.copy(x0)
	r = applyLinearOperator(x_base, x0) - b
	rho = np.linalg.norm(r)
	beta = rho
	b_norm = np.linalg.norm(b)
	
	Q = np.zeros((x0.size, kmax+1))
	H = np.zeros((kmax+1, kmax))
	
	min_error = np.inf
	min_vector = np.zeros(x0.size)
	
	Q[:,0] = r/np.linalg.norm(r)
	for k in range(1, kmax):
		Q[:,k], H[:k+1,k-1] = arnoldi_iteration_inner(x_base, Q[:,0:k], k)
		res = Hookstep(H, beta, k, tr)
		rho = np.linalg.norm(res.fun)
		if rho < min_error:
			min_error = rho
			xk = np.matmul(Q[:,0:k], res.x)
			
	test = np.linalg.norm(applyNonLinearOperator(np.copy(x_base)+x0+xk))
	tr_local = tr
	while test > 0.99*b_norm and tr_local > 1e-10:
		res = Hookstep(H, beta, kmax - 1, tr_local)
		xk = np.matmul(Q[:,0:(kmax-1)], res.x)
		min_error = np.linalg.norm(res.fun)
		test = np.linalg.norm(applyNonLinearOperator(np.copy(x_base)+x0+xk))
		tr_local = 0.5*tr_local
	return x0 + xk, min_error, tr_local

error = 0
b = applyNonLinearOperator(f)
final_error = np.linalg.norm(b)/np.linalg.norm(f)
newton_iter = 0
while newton_iter < newton_iterations:
	if np.linalg.norm(b) < np.linalg.norm(f)*tolerance:
		print('Tolerance threshold reached.')
		log_file.writelines('Tolerance threshold reached.' + '\n')
		final_error = np.linalg.norm(b)/np.linalg.norm(f)
		break
	if newton_iter == 0:
		print("Initial error = " + str(np.linalg.norm(b)/np.linalg.norm(f))  + ",    T = " + str(tfac*f[0])+ ",    d = " + str(dfac*f[1]))
		log_file.writelines("Initial error = " + str(np.linalg.norm(b)/np.linalg.norm(f))  + ",    T = " + str(tfac*f[0]) + ",    d = " + str(dfac*f[1]) + '\n')
	phi_base = phi(f[2:], f[0]*tfac, f[1]*dfac)
	zn, error, tr = GMRES(f, z0, b, krylov_dim, trust_radius)
	f = f + zn
	b = applyNonLinearOperator(f)
	final_error = np.linalg.norm(b)/np.linalg.norm(f)
	newton_iter = newton_iter + 1
	print("Iteration = " + str(newton_iter) + ",  " + "error = " + str(np.linalg.norm(b)/np.linalg.norm(f)) + ",    T = " + str(tfac*f[0])+ ",    d = " + str(dfac*f[1]) + ",    Linear system error = " + str(error) + ",    trust radius = " + str(tr))
	log_file.writelines("Iteration = " + str(newton_iter) + ",  " + "error = " + str(np.linalg.norm(b)/np.linalg.norm(f)) + ",    T = " + str(tfac*f[0])+ ",    d = " + str(dfac*f[1]) + ",    trust radius = " + str(tr) + '\n')
	if timeit.default_timer() - start > time_limit*3600:
		print('Time limit reached.')
		log_file.writelines('Time limit reached.' + '\n')
		final_error = np.linalg.norm(b)/np.linalg.norm(f)
		break

print("Iteration = " + str(newton_iter) + ",  " + "error = " + str(final_error) + ",    T = " + str(tfac*f[0])+ ",    d = " + str(dfac*f[1]) + ",    Linear system error = " + str(error) + ",    trust radius = " + str(trust_radius))
log_file.writelines("Iteration = " + str(newton_iter) + ",  " + "error = " + str(final_error) + ",    T = " + str(tfac*f[0])+ ",    d = " + str(dfac*f[1]) + ",    Linear system error = " + str(error) + ",    trust radius = " + str(trust_radius) + '\n')

runtime = timeit.default_timer() - start
print('Newton solver runtime = ' + str(runtime))
log_file.writelines('Newton solver runtime = ' + str(runtime) + '\n')

QAA_field_out = domain.new_field()
QAB_field_out = domain.new_field()	
U_field_out = domain.new_field()	
V_field_out = domain.new_field()	

QAA_field_out['c'] = VectorToGrid(f[qaa_begin+2:qaa_end+2])
QAB_field_out['c'] = VectorToGrid(f[qab_begin+2:qab_end+2])
U_field_out['c'] = VectorToGrid(f[u_begin+2:u_end+2])
V_field_out['c'] = VectorToGrid(f[v_begin+2:v_end+2])

QAA_field_out.require_coeff_space()
QAB_field_out.require_coeff_space()
U_field_out.require_coeff_space()
V_field_out.require_coeff_space()

QAA_coeff_data_out = np.copy(QAA_field_out.data)
QAB_coeff_data_out = np.copy(QAB_field_out.data)
U_coeff_data_out = np.copy(U_field_out.data)
V_coeff_data_out = np.copy(V_field_out.data)

fields_data = [np.copy(QAA_coeff_data_out), np.copy(QAB_coeff_data_out), np.copy(U_coeff_data_out), np.copy(V_coeff_data_out)]

h5f = h5py.File('solution.h5', 'w')
#------------ Simulation parameters -------------------- #
h5f.create_dataset('/ECS_id', data = ECS_id)
h5f.create_dataset('/params/Ra', data = Ra)
h5f.create_dataset('/params/Re', data = Re)
h5f.create_dataset('/params/Er', data = Er)
h5f.create_dataset('/params/lambda', data = 0)
h5f.create_dataset('/params/height', data = height)
h5f.create_dataset('/params/width', data = width)
h5f.create_dataset('/params/NX', data = NX)
h5f.create_dataset('/params/NY', data = NY)
h5f.create_dataset('/params/dt', data = dt_nominal)
h5f.create_dataset('/params/krylov_dim', data = krylov_dim)
h5f.create_dataset('T', data = tfac*f[0])
h5f.create_dataset('d', data = dfac*f[1])
h5f.create_dataset('res', data = final_error)
#------------ Coefficient space data -------------------- #
h5f.create_dataset('QAA_coeff', data = QAA_coeff_data_out)
h5f.create_dataset('QAB_coeff', data = QAB_coeff_data_out)
h5f.create_dataset('U_coeff', data = U_coeff_data_out)
h5f.create_dataset('V_coeff', data = V_coeff_data_out)
#------------ Grid space data, including derivates for calculating vorticity -------------------- #
Uy_field_out = U_field_out.differentiate(y = 1)
Vx_field_out = V_field_out.differentiate(x = 1)

U_field_out.set_scales((1,1))
V_field_out.set_scales((1,1))
Uy_field_out.set_scales((1,1))
Vx_field_out.set_scales((1,1))

QAA_field_out.require_grid_space()
QAB_field_out.require_grid_space()
U_field_out.require_grid_space()
V_field_out.require_grid_space()
Uy_field_out.require_grid_space()
Vx_field_out.require_grid_space()

h5f.create_dataset('x', data = xg)
h5f.create_dataset('y', data = yg)
h5f.create_dataset('QAA', data = QAA_field_out.data)
h5f.create_dataset('QAB', data = QAB_field_out.data)
h5f.create_dataset('U', data = U_field_out.data)
h5f.create_dataset('V', data = V_field_out.data)
h5f.create_dataset('Uy', data = Uy_field_out.data)
h5f.create_dataset('Vx', data = Vx_field_out.data)
#--------------------------------------------------------------------- #


#--------------------------------------------------------------------- #
# Check for symmetries by computing the residual ||S*f - f||, 
# where S is the symmetry operator and f is the state.
#--------------------------------------------------------------------- #
def TranslateTest(fields_data, n):
	shift = width/n
	fields_data_t = [np.copy(fields_data[0]), np.copy(fields_data[1]), np.copy(fields_data[2]), np.copy(fields_data[3])]
	for k in range(4):
		for i in range(NXH):
			fields_data_t[k][i,:] = np.exp(2*1j*i*np.pi*(shift/width))*fields_data_t[k][i,:]
	D = 0
	for k in range(4):
		D = D + np.linalg.norm(fields_data_t[k] - fields_data[k])
	return D
	
def S1TranslateTest(fields_data, n):
	shift = width/n
	fields_data_t = [np.copy(fields_data[0]), np.copy(fields_data[1]), np.copy(fields_data[2]), np.copy(fields_data[3])]
	
	for k in [0,2]:
		for j in range(NY):
			if j % 2 == 0:
				fields_data_t[k][:,j] = fields_data_t[k][:,j]
			elif j % 2 == 1:
				fields_data_t[k][:,j] = -fields_data_t[k][:,j]
				
	for k in [1,3]:
		for j in range(NY):
			if j % 2 == 0:
				fields_data_t[k][:,j] = -fields_data_t[k][:,j]
			elif j % 2 == 1:
				fields_data_t[k][:,j] = fields_data_t[k][:,j]
	
	for k in range(4):
		for i in range(NXH):
			fields_data_t[k][i,:] = np.exp(2*1j*i*np.pi*(shift/width))*fields_data_t[k][i,:]
	D = 0
	for k in range(4):
		D = D + np.linalg.norm(fields_data_t[k] - fields_data[k])
	return D
	
def S2TranslateTest(fields_data, n):
	shift = width/n
	fields_data_t = [np.copy(fields_data[0]), np.copy(fields_data[1]), np.copy(fields_data[2]), np.copy(fields_data[3])]
	
	for k in [0,3]:
		for i in range(NXH):
			fields_data_t[k][i,:] = np.conj(fields_data_t[k][i,:])
				
	for k in [1,2]:
		for i in range(NXH):
			fields_data_t[k][i,:] = -np.conj(fields_data_t[k][i,:])

	for k in range(4):
		for i in range(NXH):
			fields_data_t[k][i,:] = np.exp(2*1j*i*np.pi*(shift/width))*fields_data_t[k][i,:]
	D = 0
	for k in range(4):
		D = D + np.linalg.norm(fields_data_t[k] - fields_data[k])
	return D

for i in range(2, 13):
	h5f.create_dataset('symmetries/T' + str(i), data = TranslateTest(fields_data, i))

h5f.create_dataset('symmetries/S1T2', data = S1TranslateTest(fields_data, 2))
h5f.create_dataset('symmetries/S2T2', data = S2TranslateTest(fields_data, 2))
#--------------------------------------------------------------------- #

h5f.close()

T_file = open('T.txt', 'w')
T_file.writelines(str(tfac*f[0]))
T_file.close()

if final_error < tolerance:
	print('Solver converged with tolerance ' + str(final_error))
	log_file.writelines('Solver converged with tolerance ' + str(final_error) + '\n')

stop = timeit.default_timer()
print('Time: ' + str(stop-start) + ' seconds')
log_file.writelines('Time: ' + str(stop-start) + ' seconds')
log_file.close()

if not os.path.exists('./newton_iterations'):
	os.mkdir('./newton_iterations')

index = 1
while os.path.exists('./newton_iterations/' + str(index)):
	index = index + 1
os.mkdir('./newton_iterations/' + str(index))

for filename in glob.glob(os.path.join('./', '*.*')):
    shutil.copy(filename, './newton_iterations/' + str(index))

# Output data over a single period of the ECS
def phi_out(array_in, T):
	x_basis_phi = de.Fourier('x', NX, interval=(-0.5*width, 0.5*width), dealias = dealias_fac)
	y_basis_phi = de.Chebyshev('y', NY, interval=(0, height), dealias = dealias_fac)
	domain_phi = de.Domain([x_basis_phi, y_basis_phi], grid_dtype = np.float64)
	problem_phi = de.IVP(domain_phi, variables=['p','U','V','Ux','Uy','Vx','Vy','QAA','QAAx','QAAy','QAB','QABx','QABy'])
	problem_phi.meta[:]['y']['dirichlet'] = True
	problem_phi.parameters['width'] = width
	problem_phi.parameters['height'] = height
	problem_phi.parameters['RaEr'] = Ra/Er
	problem_phi.parameters['Re'] = Re
	problem_phi.add_equation("Ux - dx(U) = 0")
	problem_phi.add_equation("Uy - dy(U) = 0")
	problem_phi.add_equation("Vx - dx(V) = 0")
	problem_phi.add_equation("Vy - dy(V) = 0")
	problem_phi.add_equation("QAAx - dx(QAA) = 0")
	problem_phi.add_equation("QAAy - dy(QAA) = 0")
	problem_phi.add_equation("QABx - dx(QAB) = 0")
	problem_phi.add_equation("QABy - dy(QAB) = 0")
	problem_phi.add_equation("dt(QAA)-dx(QAAx)-dy(QAAy)-QAA=-10*QAA*QAA*QAA-10*QAA*QAB*QAB-QAAx*U-QAAy*V+QAB*Uy-QAB*Vx")
	problem_phi.add_equation("dt(QAB)-dx(QABx)-dy(QABy)-QAB=-10*QAA*QAA*QAB-10*QAB*QAB*QAB-QAA*Uy+QAA*Vx-QABx*U-QABy*V")
	problem_phi.add_equation("Re*dt(U)+dx(p)-2*dx(Ux)-dy(Uy)-dx(Vy)+RaEr*QAAx+RaEr*QABy=-Re*U*Ux-Re*Uy*V")
	problem_phi.add_equation("Re*dt(V)+dy(p)-2*dy(Vy)-dx(Uy)-dx(Vx)-RaEr*QAAy+RaEr*QABx=-Re*U*Vx-Re*V*Vy")
	problem_phi.add_equation("Ux + Vy = 0")
	problem_phi.add_bc("left(U) = 0")
	problem_phi.add_bc("left(V) = 0")
	problem_phi.add_bc("left(QAA) = -0.5")
	problem_phi.add_bc("left(QAB) = 0")
	problem_phi.add_bc("right(U) = 0")
	problem_phi.add_bc("right(V) = 0", condition="(nx != 0)")
	problem_phi.add_bc("left(p) = 0", condition="(nx == 0)")
	problem_phi.add_bc("right(QAA) = -0.5")
	problem_phi.add_bc("right(QAB) = 0")
	solver_phi = problem_phi.build_solver(timestepper)
	
	QAA = solver_phi.state['QAA']
	QAB = solver_phi.state['QAB']
	U = solver_phi.state['U']
	V = solver_phi.state['V']
	
	QAA['c'] = VectorToGrid(array_in[qaa_begin:qaa_end])
	QAB['c'] = VectorToGrid(array_in[qab_begin:qab_end])
	U['c'] = VectorToGrid(array_in[u_begin:u_end])
	V['c'] = VectorToGrid(array_in[v_begin:v_end])
	
	order_params = solver_phi.evaluator.add_file_handler('order_params', iter = 1, max_size = np.inf)
	order_params.add_task("integ(U)/(height*width)", layout='g', name='u_int')
	order_params.add_task("integ(V*V)/(height*width)", layout='g', name='v2_int')
	order_params.add_task("integ(QAA)/(height*width)", layout='g', name='qaa_int')
	
	if output_full_trajectory == 1:
		full_solution = solver_phi.evaluator.add_file_handler('full_solution', iter = 20, max_size = np.inf)
		full_solution.add_task('QAA', layout='g', name='QAA')
		full_solution.add_task('QAB', layout='g', name='QAB')
		full_solution.add_task('U', layout='g', name='U')
		full_solution.add_task('V', layout='g', name='V')
		full_solution.add_task('Uy', layout='g', name='Uy')
		full_solution.add_task('Vx', layout='g', name='Vx')
		full_solution.add_task('QAA', layout='c', name='QAA_coeff')
		full_solution.add_task('QAB', layout='c', name='QAB_coeff')
		full_solution.add_task('U', layout='c', name='U_coeff')
		full_solution.add_task('V', layout='c', name='V_coeff')

	num_steps = int(T/dt_nominal)
	dt = T/num_steps
	for i in range(num_steps):
		solver_phi.step(dt)

if final_error < tolerance:
	if not os.path.exists('./solution'):
		os.mkdir('./solution')
	for filename in glob.glob(os.path.join('./', '*.*')):
		shutil.copy(filename, './solution')
	if not os.path.exists('./stability'):
		os.mkdir('./stability')
	
	phi_out(f[2:], f[0]*tfac)
	if not os.path.exists('./time-dependent'):
		os.mkdir('./time-dependent')
	if os.path.exists('./full_solution'):
		shutil.move('./full_solution', './time-dependent/full_solution')
	if os.path.exists('./order_params'):
		shutil.move('./order_params', './time-dependent/order_params')
		
	phi_base = phi(f[2:], f[0]*tfac, f[1]*dfac)
	Q, H_ = arnoldi_iteration(f[2:], f[0]*tfac, f[1]*dfac, np.random.rand(MY), 60)
	H = H_[0:-1,:]

	scipy.io.mmwrite('./stability/H.mtx', H) # Hessenberg matrix
	w, vr_ = scipy.linalg.eig(H)
	w_abs = np.abs(w)
	vr = np.matmul(Q[:,0:-1], vr_)
	scipy.io.mmwrite('./stability/w.mtx', [w]) # Eigenvalues; .mtx = Matrix Market format

	# Use a threshold of '1.01' to get a quick estimate of the number of unstable directions.
	# A threshold greater than 1 is used because the marginal directions sometimes have
	# eigenvalue with abs value slightly greater than 1, due to numerical error.
	counter = 0
	for i in range(w.size):
		if w_abs[i] > 1.01:
			counter = counter + 1

	outputFile = open('./stability/N_unstable.txt', 'w')
	outputFile.writelines(str(counter))
	outputFile.close()

	x_basis_out = de.Fourier('x', NX, interval = (-0.5*width, 0.5*width), dealias = dealias_fac)
	y_basis_out = de.Chebyshev('y', NY, interval = (0, height), dealias = dealias_fac)
	domain_out = de.Domain([x_basis_out, y_basis_out], grid_dtype=np.complex128)
	xg_out = domain_out.grid(0)
	yg_out = domain_out.grid(1)

	def VectorToCGrid(array_in):
		field_temp = domain_out.new_field()
		vtemp = np.zeros((NX-1, NY), dtype = np.complex128)
		vtemp[0,:] = array_in[0:NY].real
		for i in range(1, NXH):
			ih = NXH+i
			CR = array_in[(i*NY):(i+1)*NY].real
			CI = array_in[(i*NY):(i+1)*NY].imag
			SR = array_in[(ih*NY):(ih+1)*NY].real
			SI = array_in[(ih*NY):(ih+1)*NY].imag
			a = 0.5*(CR + SI)
			b = 0.5*(CI - SR)
			c = 0.5*(CR - SI)
			d = 0.5*(CI + SR)
			vtemp[i,:] = a + 1j*b
			vtemp[NX-i-1,:] = c + 1j*d
		
		field_temp['c'] = np.copy(vtemp)
		field_temp.require_grid_space()
		return np.copy(field_temp.data)
		
	Evals = np.zeros(0, dtype = np.complex128)
	Evecs_grid = np.zeros((0, 4, NX, NY), dtype = np.complex128)
	Evecs_coeff = np.zeros((0, MY), dtype = np.complex128)
	for i in range(w.size):
		if w_abs[i] > 1:
			Evals = np.append(Evals, w[i])
			Evecs_temp1 = np.zeros((1, MY), dtype = np.complex128)
			Evecs_temp2 = np.zeros((1, 4, NX, NY), dtype = np.complex128)
			Evecs_coeff = np.append(Evecs_coeff, Evecs_temp1, axis = 0)
			EV_temp = np.copy(vr[:,i])
			Evecs_temp2[0,0,:,:] = VectorToCGrid(EV_temp[qaa_begin:qaa_end])
			Evecs_temp2[0,1,:,:] = VectorToCGrid(EV_temp[qab_begin:qab_end])
			Evecs_temp2[0,2,:,:] = VectorToCGrid(EV_temp[u_begin:u_end])
			Evecs_temp2[0,3,:,:] = VectorToCGrid(EV_temp[v_begin:v_end])
			Evecs_grid = np.append(Evecs_grid, Evecs_temp2, axis = 0)

	h5f = h5py.File('./stability/vu.h5', 'w')
	h5f.create_dataset('/x', data = xg)
	h5f.create_dataset('/y', data = yg)
	h5f.create_dataset('/eigenvalues', data = Evals)
	h5f.create_dataset('/eigenvectors_coeff', data = Evecs_coeff)
	h5f.create_dataset('/eigenvectors_grid', data = Evecs_grid)
	h5f.close()
	
	files = [f_ for f_ in os.listdir('.') if os.path.isfile(f_)]
	for filename in files:
		os.remove(filename)

print('Time: ',  timeit.default_timer() - start)










