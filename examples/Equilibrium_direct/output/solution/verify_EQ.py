'''
Computes the residual of the UNpreconditioned nonlinear operator F(X), which
case be used an as extra verification for equilibria.
Due to amplification of errors when calculating derivatives, the residual may be larger
than expected, even for accurate input states. Generally, high grid resolution is required
to see the direct residual approach machine precision.
'''

import h5py
import scipy.io
from scipy import sparse
from scipy import optimize
from numpy.polynomial import polynomial as P

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
# Input file
#---------------------------------------- #
input_filename = './solution.h5'
input_file = h5py.File(input_filename, 'r')
#---------------------------------------- #

#---------------------------------------- #
# Log file
#---------------------------------------- #
log_file = open('verification.out', 'w')
#---------------------------------------- #

#---------------------------------------- #
# Parameters
#---------------------------------------- #
d_input_mode = 'c'

Ra = 1.0*np.array(input_file.get('/params/Ra'))

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

n_fields = 5 # Number of physical fields. Here, the pressure is included, so there are 5, in addition to (QAA, QAB, U, V).
MY = n_fields*NX*NY # Total number of degrees of freedom

Re = 0.0136
Er = 1.0
RaEr = Ra/Er
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
# Initial data input
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
log_file.writelines('---------------------------------' + '\n')
log_file.writelines('\n')

def CalculatePressure(array_in):
	x_basisL = de.Fourier('x', NX, interval=(-0.5*width, 0.5*width), dealias = dealias_fac)
	y_basisL = de.Chebyshev('y', NY, interval=(0, height), dealias = dealias_fac)
	domainL = de.Domain([x_basisL, y_basisL], grid_dtype=np.float64)
	
	array_temp = np.copy(array_in)
	
	QAA = domainL.new_field()
	QAA['c'] = VectorToGrid(array_temp[qaa_begin:qaa_end])
	QAAx = QAA.differentiate(x = 1)
	QAAxx = QAA.differentiate(x = 2)
	QAAy = QAA.differentiate(y = 1)
	QAAyy = QAA.differentiate(y = 2)
	QAAxy = QAA.differentiate(x = 1, y = 1)

	QAB = domainL.new_field()
	QAB['c'] = VectorToGrid(array_temp[qab_begin:qab_end])
	QABx = QAB.differentiate(x = 1)
	QABxx = QAB.differentiate(x = 2)
	QABy = QAB.differentiate(y = 1)
	QAByy = QAB.differentiate(y = 2)
	QABxy = QAB.differentiate(x = 1, y = 1)

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
	
	QAA_op = (QAAxx+QAAyy+QAA-10*QAA*QAA*QAA-10*QAA*QAB*QAB-QAAx*U-QAAy*V+QAB*Uy-QAB*Vx).evaluate()
	QAB_op = (QABxx+QAByy+QAB-10*QAA*QAA*QAB-10*QAB*QAB*QAB-QAA*Uy+QAA*Vx-QABx*U-QABy*V).evaluate()
	U_op = (2*Uxx+Uyy+Vxy-RaEr*QAAx-RaEr*QABy-Re*U*Ux-Re*Uy*V).evaluate()
	V_op = (2*Vyy+Uxy+Vxx+RaEr*QAAy-RaEr*QABx-Re*U*Vx-Re*V*Vy).evaluate()
	
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
	
	p = solverL.state['p']

	p.require_coeff_space()
	
	return GridToVector(np.copy(p.data))

def applyNonLinearOperator(array_in):
	x_basisL = de.Fourier('x', NX, interval=(-0.5*width, 0.5*width), dealias = dealias_fac)
	y_basisL = de.Chebyshev('y', NY, interval=(0, height), dealias = dealias_fac)
	domainL = de.Domain([x_basisL, y_basisL], grid_dtype=np.float64)
	
	array_temp = np.copy(array_in)
	
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
	
	p = domainL.new_field()
	p['c'] = VectorToGrid(array_temp[p_begin:p_end])
	px = p.differentiate(x = 1)
	py = p.differentiate(y = 1)
	
	QAA_op = (QAAxx+QAAyy+QAA-10*QAA*QAA*QAA-10*QAA*QAB*QAB-QAAx*U-QAAy*V+QAB*Uy-QAB*Vx).evaluate()
	QAB_op = (QABxx+QAByy+QAB-10*QAA*QAA*QAB-10*QAB*QAB*QAB-QAA*Uy+QAA*Vx-QABx*U-QABy*V).evaluate()
	U_op = (-px+2*Uxx+Uyy+Vxy-RaEr*QAAx-RaEr*QABy-Re*U*Ux-Re*Uy*V).evaluate()
	V_op = (-py+2*Vyy+Uxy+Vxx+RaEr*QAAy-RaEr*QABx-Re*U*Vx-Re*V*Vy).evaluate()
	p_op = (Ux+Vy).evaluate()
	
	QAA_op.require_coeff_space()
	QAB_op.require_coeff_space()
	U_op.require_coeff_space()
	V_op.require_coeff_space()
	p_op.require_coeff_space()
	
	array_out = np.zeros(MY)
	array_out[qaa_begin:qaa_end] = np.copy(GridToVector(QAA_op.data))
	array_out[qab_begin:qab_end] = np.copy(GridToVector(QAB_op.data))
	array_out[u_begin:u_end] = np.copy(GridToVector(U_op.data))
	array_out[v_begin:v_end] = np.copy(GridToVector(V_op.data))
	array_out[p_begin:p_end] = np.copy(GridToVector(p_op.data))
	return array_out

f[p_begin:p_end] = CalculatePressure(f[0:v_end])
b = applyNonLinearOperator(f)
print('Direct residual = ' + str(np.linalg.norm(b)))