import h5py
import scipy.io
from scipy import sparse
from scipy import optimize

import numpy as np
from mpi4py import MPI
from dedalus import public as de
from dedalus.extras import flow_tools

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
# Solver parameters. 
# Note that flow alignment is not included in the current version of this script.
#---------------------------------------- #
ECS_id = '' # (optional) identifier for ECS
d_input_mode = 'c' # Data input format; 'c' = coefficient representation and 'g' = grid representation
output_full_trajectory = 0 # After the Newton solver converges, the code automatically time integrates over a single period and outputs
                                       # time series of some channel averages. There is also the option to output the full field data by setting 'output_full_trajectory' to 1

Ra = 1.0*np.array(input_file.get('/params/Ra'))
time_limit = 6
tolerance = 1e-10 # Convergence criterion
newton_iterations = 15
krylov_dim = 80 # Maximum krylov subspace dimension
kmin = 40 # Minimum krylov subspace dimension
kfreq = 20 # Increment between 'kmin' and 'krylov_dim'
trust_radius = 0.1
tr_min = 1e-8 # Minimum trust radius to try

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
dealias_fac = 2
x_basis = de.Fourier('x', NX, interval=(-0.5*width, 0.5*width), dealias = dealias_fac)
y_basis = de.Chebyshev('y', NY, interval=(0, height), dealias = dealias_fac)
domain = de.Domain([x_basis, y_basis], grid_dtype = np.float64)
xg = domain.grid(0)
yg = domain.grid(1)
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

Re = 0.0136
Er = 1.0
RaEr = Ra/Er
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
f = np.zeros(MY)
qaa_begin = 0
qaa_end = qaa_begin + (NX*NY)
qab_begin = qaa_end
qab_end = qab_begin + (NX*NY)
u_begin = qab_end
u_end = u_begin + (NX*NY)
v_begin = u_end
v_end = v_begin + (NX*NY)
p_begin = v_end
p_end = p_begin + (NX*NY)
f[qaa_begin:qaa_end] = GridToVector(np.copy(QAA_data_init_coeff))
f[qab_begin:qab_end] = GridToVector(np.copy(QAB_data_init_coeff))
f[u_begin:u_end] = GridToVector(np.copy(U_data_init_coeff))
f[v_begin:v_end] = GridToVector(np.copy(V_data_init_coeff))

# Initial iterate for GMRES; use later.
z0 = np.zeros(MY)
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
log_file.writelines('trust radius = ' + str(trust_radius) + '\n')
log_file.writelines('---------------------------------' + '\n')
log_file.writelines('\n')

''' Function applyLinearOperator:

	Description
	---------------------
	Compute the preconditioned linear operator (nabla)^{-2} * L_{OP}
	---------------------

	Parameters
	---------------------
	array_base :
		variable type: Real-valued, 1d numpy data array with dimension MY = n_fields*NX*NY
		description: initial (base) vector x0 in coefficient format
	array_pert :
		variable type: Real-valued, 1d numpy data array with dimension MY = n_fields*NX*NY
		description: 'delta_x' in the equation for the Matrix-vector product
	---------------------
'''
def applyLinearOperator(array_base, array_pert):
	x_base = np.copy(array_base)
	delta_x = np.copy(array_pert)
	
	x_basisL = de.Fourier('x', NX, interval=(-0.5*width, 0.5*width), dealias = dealias_fac)
	y_basisL = de.Chebyshev('y', NY, interval=(0, height), dealias = dealias_fac)
	domainL = de.Domain([x_basisL, y_basisL], grid_dtype = np.float64)
	
	#---------------------------------------- #
	# Compute the field representation of the base state, including any required derivatives.
	# Note: the	'eq' postscript is potentially confusing as it coincides with the abbreviation for equilibria.
	# It will be removed in later releases.
	#---------------------------------------- #
	QAAeq = domainL.new_field()
	QAAeq['c'] = VectorToGrid(x_base[qaa_begin:qaa_end])
	QAAeqx = QAAeq.differentiate(x = 1)
	QAAeqy = QAAeq.differentiate(y = 1)

	QABeq = domainL.new_field()
	QABeq['c'] = VectorToGrid(x_base[qab_begin:qab_end])
	QABeqx = QABeq.differentiate(x = 1)
	QABeqy = QABeq.differentiate(y = 1)

	Ueq = domainL.new_field()
	Ueq['c'] = VectorToGrid(x_base[u_begin:u_end])
	Ueqx = Ueq.differentiate(x = 1)
	Ueqy = Ueq.differentiate(y = 1)
	
	Veq = domainL.new_field()
	Veq['c'] = VectorToGrid(x_base[v_begin:v_end])
	Veqx = Veq.differentiate(x = 1)
	Veqy = Veq.differentiate(y = 1)
	#---------------------------------------- #
	
	#---------------------------------------- #
	# Compute the field representation of the perturbation, including any required derivatives.
	#---------------------------------------- #
	QAA = domainL.new_field()
	QAA['c'] = VectorToGrid(delta_x[qaa_begin:qaa_end])
	QAAx = QAA.differentiate(x = 1)
	QAAxx = QAA.differentiate(x = 2)
	QAAy = QAA.differentiate(y = 1)
	QAAyy = QAA.differentiate(y = 2)

	QAB = domainL.new_field()
	QAB['c'] = VectorToGrid(delta_x[qab_begin:qab_end])
	QABx = QAB.differentiate(x = 1)
	QABxx = QAB.differentiate(x = 2)
	QABy = QAB.differentiate(y = 1)
	QAByy = QAB.differentiate(y = 2)

	U = domainL.new_field()
	U['c'] = VectorToGrid(delta_x[u_begin:u_end])
	Ux = U.differentiate(x = 1)
	Uxx = U.differentiate(x = 2)
	Uy = U.differentiate(y = 1)
	Uyy = U.differentiate(y = 2)
	Uxy = U.differentiate(x = 1, y = 1)
	
	V = domainL.new_field()
	V['c'] = VectorToGrid(delta_x[v_begin:v_end])
	Vx = V.differentiate(x = 1)
	Vxx = V.differentiate(x = 2)
	Vy = V.differentiate(y = 1)
	Vyy = V.differentiate(y = 2)
	Vxy = V.differentiate(x = 1, y = 1)
	#---------------------------------------- #
	
	#---------------------------------------- #
	# Un-preconditioned linear operator, calculated using the analytical expression derived
	# from linearization.
	#---------------------------------------- #
	QAA_op = (-(QAAxx+QAAyy+QAA-30*QAA*QAAeq*QAAeq-10*QAA*QABeq*QABeq-20*QAAeq*QAB*QABeq-QAAeqx*U-QAAeqy*V-QAAx*Ueq-QAAy*Veq+QAB*Ueqy-QAB*Veqx+QABeq*Uy-QABeq*Vx)).evaluate()
	QAB_op = (-(QABxx+QAByy+QAB-20*QAA*QAAeq*QABeq-10*QAAeq*QAAeq*QAB-30*QAB*QABeq*QABeq-QAA*Ueqy+QAA*Veqx-QAAeq*Uy+QAAeq*Vx-QABeqx*U-QABeqy*V-QABx*Ueq-QABy*Veq)).evaluate()
	U_op = (-(+2*Uxx+Uyy+Vxy-RaEr*QAAx-RaEr*QABy-Re*U*Ueqx-Re*Ueq*Ux-Re*Ueqy*V-Re*Uy*Veq)).evaluate()
	V_op = (-(Uxy+Vxx+2*Vyy+RaEr*QAAy-RaEr*QABx-Re*U*Veqx-Re*Ueq*Vx-Re*V*Veqy-Re*Veq*Vy)).evaluate()
	#---------------------------------------- #

	#---------------------------------------- #
	# Apply inverse Laplacian preconditioner by solving the corresponding linear boundary value problem (LBVP)
	#---------------------------------------- #
	problemL = de.LBVP(domainL, variables=['p','U','V','Ux','Uy','Vx','Vy','QAA','QAAx','QAAy','QAB','QABx','QABy'])
	problemL.meta[:]['y']['dirichlet'] = True
	
	problemL.parameters['QAA_rhs'] = QAA_op
	problemL.parameters['QAB_rhs'] = QAB_op
	problemL.parameters['U_rhs'] = U_op
	problemL.parameters['V_rhs'] = V_op
	problemL.parameters['Re'] = Re
	
	problemL.add_equation("Ux - dx(U) = 0")
	problemL.add_equation("Uy - dy(U) = 0")
	problemL.add_equation("Vx - dx(V) = 0")
	problemL.add_equation("Vy - dy(V) = 0")
	problemL.add_equation("QAAx - dx(QAA) = 0")
	problemL.add_equation("QAAy - dy(QAA) = 0")
	problemL.add_equation("QABx - dx(QAB) = 0")
	problemL.add_equation("QABy - dy(QAB) = 0")
	problemL.add_equation("dx(QAAx)+dy(QAAy)=QAA_rhs")
	problemL.add_equation("dx(QABx)+dy(QABy)=QAB_rhs")
	problemL.add_equation("dx(Ux)+dy(Uy)+dx(p)=U_rhs")
	problemL.add_equation("dx(Vx)+dy(Vy)+dy(p)=V_rhs")
	problemL.add_equation("Ux+Vy=0")
	problemL.add_bc("left(U) = 0")
	problemL.add_bc("left(V) = 0")
	problemL.add_bc("left(QAA) = 0")
	problemL.add_bc("left(QAB) = 0")
	problemL.add_bc("right(U) = 0")
	problemL.add_bc("right(V) = 0", condition="(nx != 0)")
	problemL.add_bc("left(p) = 0", condition="(nx == 0)")
	problemL.add_bc("right(QAA) = 0")
	problemL.add_bc("right(QAB) = 0")
	
	solverL = problemL.build_solver()
	solverL.solve()
	
	QAA = solverL.state['QAA']
	QAB = solverL.state['QAB']
	U = solverL.state['U']
	V = solverL.state['V']

	QAA.require_coeff_space()
	QAB.require_coeff_space()
	U.require_coeff_space()
	V.require_coeff_space()
	#---------------------------------------- #
	
	array_out = np.zeros(MY)
	array_out[qaa_begin:qaa_end] = GridToVector(np.copy(QAA.data))
	array_out[qab_begin:qab_end] = GridToVector(np.copy(QAB.data))
	array_out[u_begin:u_end] = GridToVector(np.copy(U.data))
	array_out[v_begin:v_end] = GridToVector(np.copy(V.data))
	return array_out

''' Function applyNonLinearOperator:

	Description
	---------------------
	Compute the preconditioned nonlinear operator (nabla)^{-2} * F(X)
	---------------------

	Parameters
	---------------------
	array_in :
		variable type: Real-valued, 1d numpy data array with dimension MY = n_fields*NX*NY
		description: input vector
	---------------------
'''
def applyNonLinearOperator(array_in):
	x_basisL = de.Fourier('x', NX, interval=(-0.5*width, 0.5*width), dealias = dealias_fac)
	y_basisL = de.Chebyshev('y', NY, interval=(0, height), dealias = dealias_fac)
	domainL = de.Domain([x_basisL, y_basisL], grid_dtype=np.float64)
	array_temp = np.copy(array_in)
	
	#---------------------------------------- #
	# Compute the field representation of the input state, including any required derivatives.
	#---------------------------------------- #
	QAA = domainL.new_field()
	QAA['c'] = VectorToGrid(array_temp[qaa_begin:qaa_end])
	QAAx = QAA.differentiate(x = 1)
	QAAxx = QAA.differentiate(x = 2)
	QAAy = QAA.differentiate(y = 1)
	QAAyy = QAA.differentiate(y = 2)

	QAB = domainL.new_field()
	QAB['c'] = VectorToGrid(array_temp[qab_begin:qab_end])
	QABx = QAB.differentiate(x = 1)
	QABxx = QAB.differentiate(x = 2)
	QABy = QAB.differentiate(y = 1)
	QAByy = QAB.differentiate(y = 2)

	U = domainL.new_field()
	U['c'] = VectorToGrid(array_temp[u_begin:u_end])
	Ux = U.differentiate(x = 1)
	Uxx = U.differentiate(x = 2)
	Uy = U.differentiate(y = 1)
	Uyy = U.differentiate(y = 2)
	Uxy = U.differentiate(x = 1, y = 1)
	
	V = domainL.new_field()
	V['c'] = VectorToGrid(array_temp[v_begin:v_end])
	Vx = V.differentiate(x = 1)
	Vxx = V.differentiate(x = 2)
	Vy = V.differentiate(y = 1)
	Vyy = V.differentiate(y = 2)
	Vxy = V.differentiate(x = 1, y = 1)
	#---------------------------------------- #
	
	#---------------------------------------- #
	# Un-preconditioned nonlinear operator, calculated using the Dedalus 'operator' class
	#---------------------------------------- #
	QAA_op = (QAAxx+QAAyy+QAA-10*QAA*QAA*QAA-10*QAA*QAB*QAB-QAAx*U-QAAy*V+QAB*Uy-QAB*Vx).evaluate()
	QAB_op = (QABxx+QAByy+QAB-10*QAA*QAA*QAB-10*QAB*QAB*QAB-QAA*Uy+QAA*Vx-QABx*U-QABy*V).evaluate()
	U_op = (2*Uxx+Uyy+Vxy-RaEr*QAAx-RaEr*QABy-Re*U*Ux-Re*Uy*V).evaluate()
	V_op = (2*Vyy+Uxy+Vxx+RaEr*QAAy-RaEr*QABx-Re*U*Vx-Re*V*Vy).evaluate()
	#---------------------------------------- #
	
	#---------------------------------------- #
	# Apply inverse Laplacian preconditioner by solving the corresponding linear boundary value problem (LBVP)
	#---------------------------------------- #
	problemL = de.LBVP(domainL, variables=['p','U','V','Ux','Uy','Vx','Vy','QAA','QAAx','QAAy','QAB','QABx','QABy'])
	problemL.meta[:]['y']['dirichlet'] = True
	
	problemL.parameters['QAA_rhs'] = QAA_op
	problemL.parameters['QAB_rhs'] = QAB_op
	problemL.parameters['U_rhs'] = U_op
	problemL.parameters['V_rhs'] = V_op
	problemL.parameters['Re'] = Re
	
	problemL.add_equation("Ux - dx(U) = 0")
	problemL.add_equation("Uy - dy(U) = 0")
	problemL.add_equation("Vx - dx(V) = 0")
	problemL.add_equation("Vy - dy(V) = 0")
	problemL.add_equation("QAAx - dx(QAA) = 0")
	problemL.add_equation("QAAy - dy(QAA) = 0")
	problemL.add_equation("QABx - dx(QAB) = 0")
	problemL.add_equation("QABy - dy(QAB) = 0")
	problemL.add_equation("dx(QAAx)+dy(QAAy)=QAA_rhs")
	problemL.add_equation("dx(QABx)+dy(QABy)=QAB_rhs")
	problemL.add_equation("2*dx(Ux)+dx(Vy)+dy(Uy)+dx(p)=U_rhs")
	problemL.add_equation("dx(Vx)+2*dy(Vy)+dx(Uy)+dy(p)=V_rhs")
	problemL.add_equation("Ux+Vy=0")
	problemL.add_bc("left(U) = 0")
	problemL.add_bc("left(V) = 0")
	problemL.add_bc("left(QAA) = 0")
	problemL.add_bc("left(QAB) = 0")
	problemL.add_bc("right(U) = 0")
	problemL.add_bc("right(V) = 0", condition="(nx != 0)")
	problemL.add_bc("left(p) = 0", condition="(nx == 0)")
	problemL.add_bc("right(QAA) = 0")
	problemL.add_bc("right(QAB) = 0")
	
	solverL = problemL.build_solver()
	solverL.solve()
	
	QAA = solverL.state['QAA']
	QAB = solverL.state['QAB']
	U = solverL.state['U']
	V = solverL.state['V']

	QAA.require_coeff_space()
	QAB.require_coeff_space()
	U.require_coeff_space()
	V.require_coeff_space()
	#---------------------------------------- #
	
	array_out = np.zeros(MY)
	array_out[qaa_begin:qaa_end] = GridToVector(np.copy(QAA.data))
	array_out[qab_begin:qab_end] = GridToVector(np.copy(QAB.data))
	array_out[u_begin:u_end] = GridToVector(np.copy(U.data))
	array_out[v_begin:v_end] = GridToVector(np.copy(V.data))
	return array_out

def arnoldi_iteration_inner(x_base, Q, k:int):
	Qk = applyLinearOperator(x_base, Q[:, k - 1])
	Hk = np.zeros(k+1)
	for j in range(0, k):
		Hk[j] = np.matmul(np.conj(Q[:,j]), Qk)
		Qk = Qk- Hk[j]*Q[:,j]
	Hk[k] = np.linalg.norm(Qk)
	Qk = Qk/Hk[k]

	return Qk, Hk

# Used for the Hookstep; will be documented in detail in a later release
def Gmin(xb_, x0_, k_, beta_, tr_, Q_, H_):
	e1 = np.zeros(k_+1)
	e1[0] = beta_
	
	def fun(x_, F):
		r = np.matmul(F, x_) + e1
		return np.matmul(r, r)

	def Jacobian(x_, F):
		return 2*np.matmul(np.matmul(np.transpose(F), F), x_) + 2*np.matmul(np.transpose(F), e1)
	
	def constraint(x_):
		return tr_*tr_ - np.matmul(np.transpose(x_), x_)
	
	def constraintJac(x_):
		return -2*x_
	
	ineq_cons = {'type': 'ineq', 'fun' : constraint, 'jac' : constraintJac}
	
	w_init = np.zeros(k_)
	w_init[0] = 1e-3
	
	res = scipy.optimize.minimize(fun, w_init, args=(H_[0:k_+1,0:k_]), method='SLSQP', jac = Jacobian,
							constraints=(ineq_cons), options={'ftol': 1e-34, 'disp': False, 'maxiter': 10000}, bounds=None)
	xk = np.matmul(Q_[:,0:k_], res.x)
	return np.linalg.norm(applyNonLinearOperator(np.copy(xb_)+np.copy(x0_)+xk))

# Used for the Hookstep; will be documented in detail in a later release
def Gmin_full(xb_, x0_, k_, beta_, tr_, Q_, H_):
	e1 = np.zeros(k_+1)
	e1[0] = beta_
	
	def fun(x_, F):
		r = np.matmul(F, x_) + e1
		return np.matmul(r, r)

	def Jacobian(x_, F):
		return 2*np.matmul(np.matmul(np.transpose(F), F), x_) + 2*np.matmul(np.transpose(F), e1)
	
	def constraint(x_):
		return tr_*tr_ - np.matmul(np.transpose(x_), x_)
	
	def constraintJac(x_):
		return -2*x_
	
	ineq_cons = {'type': 'ineq', 'fun' : constraint, 'jac' : constraintJac}
	
	w_init = np.zeros(k_)
	w_init[0] = 1e-3
	
	res = scipy.optimize.minimize(fun, w_init, args=(H_[0:k_+1,0:k_]), method='SLSQP', jac = Jacobian,
							constraints=(ineq_cons), options={'ftol': 1e-34, 'disp': False, 'maxiter': 10000}, bounds=None)
	xk = np.matmul(Q_[:,0:k_], res.x)
	return xk, np.linalg.norm(applyNonLinearOperator(np.copy(xb_)+np.copy(x0_)+xk))

# Used for the Hookstep; will be documented in detail in a later release
def TRmin(xb, x0_, k_, beta_, tr0, Q_, H_):
	tr = np.copy(tr0)
	counter = 0
	while tr > tr_min:
		tr = 0.5*tr
		counter = counter + 1
	x_data = np.zeros(counter)
	y_data = np.zeros(counter)
	tr = np.copy(tr0)
	counter = 0
	while tr > tr_min:
		x_data[counter] = tr
		y_data[counter] = Gmin(xb, x0_, k_, beta_, tr, Q_, H_)
		counter = counter + 1
		tr = 0.5*tr
	return x_data[np.argmin(y_data)], np.min(y_data)

def GMRES(x_base, x0, b, kmax):
	x0_cached = np.copy(x0)
	r = applyLinearOperator(x_base, x0) - b
	rho = np.linalg.norm(r)
	beta = rho
	b_norm = np.linalg.norm(b)
	
	Q = np.zeros((x0.size, kmax+1))
	H = np.zeros((kmax+1, kmax))
	
	min_vector = np.copy(x0)
	MINR = np.inf
	MIN_TR = np.inf
	Q[:,0] = r/np.linalg.norm(r)
	for k in range(1, kmax+1):
		Q[:,k], H[:k+1,k-1] = arnoldi_iteration_inner(x_base, Q[:,0:k], k)

		if k > kmin and k % kfreq == 0:
			tr_min, min_res = TRmin(np.copy(x_base), x0, k, beta, trust_radius, Q, H)
			xk, error = Gmin_full(np.copy(x_base), x0, k, beta, tr_min, Q, H)
			if error < MINR:
				MINR = error
				min_vector = np.copy(xk)
				MIN_TR = tr_min

	return x0_cached + min_vector, MINR, MIN_TR






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
		print("Initial error = " + str(np.linalg.norm(b)/np.linalg.norm(f)))
		log_file.writelines("Initial error = " + str(np.linalg.norm(b)/np.linalg.norm(f)) + '\n')
	zn, error, tr = GMRES(f, z0, b, krylov_dim)
	f = f + zn
	b = applyNonLinearOperator(f)
	final_error = np.linalg.norm(b)/np.linalg.norm(f)
	newton_iter = newton_iter + 1
	print("Iteration = " + str(newton_iter) + ",  " + "error = " + str(np.linalg.norm(b)/np.linalg.norm(f)) + ",    Linear system error = " + str(error) + ",    trust radius = " + str(tr))
	log_file.writelines("Iteration = " + str(newton_iter) + ",  " + "error = " + str(np.linalg.norm(b)/np.linalg.norm(f)) + ",    Linear system error = " + str(error) + ",    trust radius = " + str(tr) + '\n')
	if timeit.default_timer() - start > time_limit*3600:
		print('Time limit reached.')
		log_file.writelines('Time limit reached.' + '\n')
		final_error = np.linalg.norm(b)/np.linalg.norm(f)
		break


print("Iteration = " + str(newton_iter) + ",  " + "error = " + str(final_error) + ",    Linear system error = " + str(error) + ",    trust radius = " + str(trust_radius))
log_file.writelines("Iteration = " + str(newton_iter) + ",  " + "error = " + str(final_error) + ",    Linear system error = " + str(error) + ",    trust radius = " + str(trust_radius) + '\n')

runtime = timeit.default_timer() - start
print('Newton solver runtime = ' + str(runtime))
log_file.writelines('Newton solver runtime = ' + str(runtime) + '\n')

QAA_field_out = domain.new_field()
QAB_field_out = domain.new_field()	
U_field_out = domain.new_field()	
V_field_out = domain.new_field()

QAA_field_out['c'] = VectorToGrid(f[qaa_begin:qaa_end])
QAB_field_out['c'] = VectorToGrid(f[qab_begin:qab_end])
U_field_out['c'] = VectorToGrid(f[u_begin:u_end])
V_field_out['c'] = VectorToGrid(f[v_begin:v_end])

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
h5f.create_dataset('/params/krylov_dim', data = krylov_dim)
h5f.create_dataset('res', data = final_error)
h5f.create_dataset('x', data = xg)
h5f.create_dataset('y', data = yg)
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
    
if final_error < tolerance:
	if not os.path.exists('./solution'):
		os.mkdir('./solution')
	for filename in glob.glob(os.path.join('./', '*.*')):
		shutil.copy(filename, './solution')












