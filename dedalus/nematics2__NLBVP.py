import h5py
import scipy.io
import os
import numpy as np
import time
import pathlib
from dedalus import public as de
import logging
logger = logging.getLogger(__name__)
import timeit
start = timeit.default_timer()

#---------------------------------------- #
# Input file
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
ECS_id = 'UNI'
d_input_mode = 'c'

Ra = 1.5

# Target tolerance for the Newton solver. 
# This is the norm of a preconditioned residual, so it is different from the "direct residual" found via substitution into the equations of motion.
# It is usually much smaller than the direct residual because the preconditioning damps out higher-order modes.
tolerance = 5e-15

# Damping for Newton iteration. Usually there isn't a significant benefit from making this smaller.
damping = 1.0

NX_input = int(1.0*np.array(input_file.get('/params/NX')))
NY_input = int(1.0*np.array(input_file.get('/params/NY')))
NXH_input = int(NX_input/2)

height = 1.0*np.array(input_file.get('/params/height'))
width = 1.0*np.array(input_file.get('/params/width'))
NX = NX_input
NY = NY_input
NXH = int(NX/2)

Re = 0.0136
Er = 1.0
lamb = 0
#---------------------------------------- #

#---------------------------------------- #
# Basis and domain
#---------------------------------------- #
dealias_fac = 2
y_basis = de.Chebyshev('y', NY, interval = (0, height), dealias = dealias_fac)
domain = de.Domain([y_basis], grid_dtype=np.float64)
yg = domain.grid(0)
#---------------------------------------- #

#---------------------------------------- #
# Initial data input
#---------------------------------------- #
if d_input_mode == 'g' and NY_input == NY:
	dataQAA_init = np.array(input_file.get('/QAA'))[0,:]
	dataQAB_init = np.array(input_file.get('/QAB'))[0,:]
	dataU_init = np.array(input_file.get('/U'))[0,:]
	dataV_init = np.array(input_file.get('/V'))[0,:]

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
	dataQAA_init = np.array(input_file.get('/QAA_coeff'))[0,:]
	dataQAB_init = np.array(input_file.get('/QAB_coeff'))[0,:]
	dataU_init = np.array(input_file.get('/U_coeff'))[0,:]
	dataV_init = np.array(input_file.get('/V_coeff'))[0,:]

	QAA_data_init_coeff = np.zeros(NY)
	QAB_data_init_coeff = np.zeros(NY)
	U_data_init_coeff = np.zeros(NY)
	V_data_init_coeff = np.zeros(NY)

	QAA_data_init_coeff[0:min(NY,NY_input)] = np.copy(dataQAA_init)[0:min(NY,NY_input)]
	QAB_data_init_coeff[0:min(NY,NY_input)] = np.copy(dataQAB_init)[0:min(NY,NY_input)]
	U_data_init_coeff[0:min(NY,NY_input)] = np.copy(dataU_init)[0:min(NY,NY_input)]
	V_data_init_coeff[0:min(NY,NY_input)] = np.copy(dataV_init)[0:min(NY,NY_input)]
else:
	if d_input_mode == 'g' and (not NY_input == NY):
		print('Error: Changing resolution on input data in grid space is not currently supported.')
	else:
		print('Error: Data input mode not recognized.')
#---------------------------------------- #


print('\n')
print('---------------------------------')
print('--- Parameters ------------------')
print('---------------------------------')
print('Ra = ' + str(Ra))
print('Er = ' + str(Er))
print('Re = ' + str(Re))
print('lamb = ' + str(lamb))
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
log_file.writelines('lamb = ' + str(lamb) + '\n')
log_file.writelines('NX = ' + str(NX) + '\n')
log_file.writelines('NY = ' + str(NY) + '\n')
log_file.writelines('height = ' + str(height) + '\n')
log_file.writelines('width = ' + str(width) + '\n')
log_file.writelines('---------------------------------' + '\n')
log_file.writelines('\n')

problem = de.NLBVP(domain, variables=['U','Uy','QAA','QAAy','QAB','QABy'])
problem.meta[:]['y']['dirichlet'] = True
problem.parameters['RaEr'] = Ra/Er
problem.parameters['Re'] = Re
problem.parameters['lamb'] = lamb
problem.add_equation("QAAy - dy(QAA) = 0")
problem.add_equation("QABy - dy(QAB) = 0")
problem.add_equation("Uy - dy(U) = 0")
problem.add_equation("-dy(QAAy)-QAA=-10*QAA*QAA*QAA-10*QAA*QAB*QAB+QAB*Uy")
problem.add_equation("-dy(QABy)-QAB-0.5*lamb*Uy=-10*QAA*QAA*QAB-10*QAB*QAB*QAB-QAA*Uy")
problem.add_equation("-dy(Uy)+RaEr*QABy=0")
problem.add_bc("left(U) = 0")
problem.add_bc("left(QAA) = -0.5")
problem.add_bc("left(QAB) = 0")
problem.add_bc("right(U) = 0")
problem.add_bc("right(QAA) = -0.5")
problem.add_bc("right(QAB) = 0")
solver = problem.build_solver()

QAA = solver.state['QAA']
QAB = solver.state['QAB']
U = solver.state['U']

QAA['c'] = np.copy(QAA_data_init_coeff)
QAB['c'] = np.copy(QAB_data_init_coeff)
U['c'] = np.copy(U_data_init_coeff)

# See Dedalus documentation for documentation on using the NLBVP solver
pert = solver.perturbations.data
pert.fill(1+tolerance)
while np.sum(np.abs(pert)) > tolerance:
    solver.newton_iteration(damping)
    print("Perturbation norm = " + str(np.sum(np.abs(pert))))

QAA_out = solver.state['QAA']
QAB_out = solver.state['QAB']
U_out = solver.state['U']

QAA_out.require_coeff_space()
QAB_out.require_coeff_space()
U_out.require_coeff_space()

#--------------------------------------------------- #
# Output solution in the full 2D
#--------------------------------------------------- #
QAA_data_2dcoeff = np.zeros((NXH, NY), dtype = np.complex128)
QAB_data_2dcoeff = np.zeros((NXH, NY), dtype = np.complex128)
U_data_2dcoeff = np.zeros((NXH, NY), dtype = np.complex128)
V_data_2dcoeff = np.zeros((NXH, NY), dtype = np.complex128)
QAA_data_2dcoeff[0,:] = np.copy(QAA.data[0:NY])
QAB_data_2dcoeff[0,:] = np.copy(QAB.data[0:NY])
U_data_2dcoeff[0,:] = np.copy(U.data[0:NY])

h5f_2d = h5py.File('./solution2d.h5', 'w')
h5f_2d.create_dataset('/params/Ra', data = Ra)
h5f_2d.create_dataset('/params/Re', data = Re)
h5f_2d.create_dataset('/params/Er', data = Er)
h5f_2d.create_dataset('/params/lambda', data = 0)
h5f_2d.create_dataset('/params/height', data = height)
h5f_2d.create_dataset('/params/width', data = width)
h5f_2d.create_dataset('/params/NX', data = NX)
h5f_2d.create_dataset('/params/NY', data = NY)
h5f_2d.create_dataset('QAA_coeff', data = QAA_data_2dcoeff)
h5f_2d.create_dataset('QAB_coeff', data = QAB_data_2dcoeff)
h5f_2d.create_dataset('U_coeff', data = U_data_2dcoeff)
h5f_2d.create_dataset('V_coeff', data = V_data_2dcoeff)
#--------------------------------------------------- #
#  ... continued below ...
#--------------------------------------------------- #

#--------------------------------------------------- #
# Output solution in 1D  
#--------------------------------------------------- #
h5f_1d = h5py.File('./solution1d.h5', 'w')
h5f_1d.create_dataset('/params/Ra', data = Ra)
h5f_1d.create_dataset('/params/Re', data = Re)
h5f_1d.create_dataset('/params/Er', data = Er)
h5f_1d.create_dataset('/params/lambda', data = 0)
h5f_1d.create_dataset('/params/height', data = height)
h5f_1d.create_dataset('/params/NY', data = NY)
h5f_1d.create_dataset('y', data = yg)
h5f_1d.create_dataset('QAA_coeff', data = np.copy(QAA_out.data))
h5f_1d.create_dataset('QAB_coeff', data = np.copy(QAB_out.data))
h5f_1d.create_dataset('U_coeff', data = np.copy(U_out.data))
#--------------------------------------------------- #
#  ... continued below ...
#--------------------------------------------------- #

#--------------------------------------------------- #
#  Calculate direction residual
#--------------------------------------------------- #
QAAy   =  QAA_out.differentiate(y = 1)
QAAyy  =  QAA_out.differentiate(y = 2)
QABy   =  QAB_out.differentiate(y = 1)
QAByy  =  QAB_out.differentiate(y = 2)
Uy   =  U_out.differentiate(y = 1)
Uyy  =  U_out.differentiate(y = 2)

QAA_nonlinear_op = (QAAyy+QAA-10*QAA*QAA*QAA-10*QAA*QAB*QAB+QAB*Uy).evaluate()
QAB_nonlinear_op = (QAByy+QAB+0.5*lamb*Uy-10*QAA*QAA*QAB-10*QAB*QAB*QAB-QAA*Uy).evaluate()
U_nonlinear_op = (-Uyy+(Ra/Er)*QABy).evaluate()

QAA_nonlinear_op.require_coeff_space()
QAB_nonlinear_op.require_coeff_space()
U_nonlinear_op.require_coeff_space()

direct_residual = np.linalg.norm(QAA_nonlinear_op.data)+np.linalg.norm(QAB_nonlinear_op.data)+np.linalg.norm(U_nonlinear_op.data)
#--------------------------------------------------- #

print('direct error = ' + str(direct_residual))
log_file.writelines('direct error = ' + str(direct_residual) + '\n')

#--------------------------------------------------- #
#  Finish file outputs
#--------------------------------------------------- #
QAA.set_scales(1)
QAB.set_scales(1)
U.set_scales(1)
QAA.require_grid_space()
QAB.require_grid_space()
U.require_grid_space()

h5f_1d.create_dataset('res', data = direct_residual)
h5f_1d.create_dataset('QAA', data = np.copy(QAA.data))
h5f_1d.create_dataset('QAB', data = np.copy(QAB.data))
h5f_1d.create_dataset('U', data = np.copy(U.data))
h5f_1d.close()

x_basis_out = de.Fourier('x', NX, interval=(-0.5*width, 0.5*width), dealias = dealias_fac)
y_basis_out = de.Chebyshev('y', NY, interval=(0, height), dealias = dealias_fac)
domain_out = de.Domain([x_basis_out, y_basis_out], grid_dtype=np.float64)
xg_out = domain_out.grid(0)
yg_out = domain_out.grid(1)

QAA_2d = domain_out.new_field()
QAB_2d = domain_out.new_field()
U_2d = domain_out.new_field()
V_2d = domain_out.new_field()
Vx_2d = domain_out.new_field()

QAA_2d['c'] = np.copy(QAA_data_2dcoeff)
QAB_2d['c'] = np.copy(QAB_data_2dcoeff)
U_2d['c'] = np.copy(U_data_2dcoeff)
V_2d['g'] = 0
Vx_2d['g'] = 0

Uy_2d = U_2d.differentiate(y = 1)

U_2d.set_scales((1,1))
V_2d.set_scales((1,1))
Uy_2d.set_scales((1,1))
Vx_2d.set_scales((1,1))

QAA_2d.require_grid_space()
QAB_2d.require_grid_space()
U_2d.require_grid_space()
V_2d.require_grid_space()
Uy_2d.require_grid_space()
Vx_2d.require_grid_space()

h5f_2d.create_dataset('res', data = direct_residual)
h5f_2d.create_dataset('x', data = xg_out)
h5f_2d.create_dataset('y', data = yg_out)
h5f_2d.create_dataset('QAA', data = QAA_2d.data)
h5f_2d.create_dataset('QAB', data = QAB_2d.data)
h5f_2d.create_dataset('U', data = U_2d.data)
h5f_2d.create_dataset('V', data = V_2d.data)
h5f_2d.create_dataset('Uy', data = Uy_2d.data)
h5f_2d.create_dataset('Vx', data = Vx_2d.data)
h5f_2d.close()
#--------------------------------------------------- #

print('Time: ' + str(timeit.default_timer() - start))
log_file.writelines('Time: ' + str(timeit.default_timer() - start) + '\n')
log_file.close()


