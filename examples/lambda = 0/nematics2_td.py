import os
import h5py # HDF5 file manipulation
import scipy.io
import numpy as np
from mpi4py import MPI
import time
import pathlib

from dedalus import public as de
from dedalus.extras import flow_tools # Use for computing CFL condition

import logging
logger = logging.getLogger(__name__)

import timeit
start = timeit.default_timer()

# Log file
log_file = open('dedalus.out', 'w')

#---------------------------------------- #
# Solver parameters
#---------------------------------------- #
d_input_mode = 'c' # Data input format; 'c' = coefficient representation and 'g' = grid representation
Ra = 4.0 # "Active" Reynolds number, equation (23) in the documentation
initial_timestep = 0.001 # Initial timestep that will be given to the adaptive timestepping routine later
max_timestep = 0.01 # Maximum value for CFL (set to np.inf if you trust the CFL or are more interested in asymptotic behavior than exact time-dependence)

Re = 0.0136 # Microscopic Reynolds number, equation (21) in the documentation
Er = 1.0 # Ericksen number, equation (22)
b = 5.0 # Free energy parameter, equation (17)
lamb = 0 # Flow alignment; note that it is 'lamb' rather than 'lambda' to avoid conflict with the reserved Python keyword of the same name

# Resolution of input data. Note that for real-valued domains, Dedalus only stores half the number of Fourier modes; 
# with the remaining half determined by complex conjugation.
NX_input = 128 # Fourier modes
NY_input = 32 # Chebyshev modes
NXH_input = int(NX_input/2)

height = 20.0 # Channel height
width = 80.0 # Channel width
NX = NX_input # Change these if needed
NY = NY_input #
NXH = int(NX/2)
#---------------------------------------- #

#---------------------------------------- #
# Initial data input
#---------------------------------------- #
input_filename = './initial_data.h5'
input_file = h5py.File(input_filename, 'r')

if d_input_mode == 'g' and NX_input == NX and NY_input == NY:
	dataQAA_init = np.array(input_file.get('/QAA'))
	dataQAB_init = np.array(input_file.get('/QAB'))
	dataU_init = np.array(input_file.get('/U'))
	dataV_init = np.array(input_file.get('/V'))

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
	
	dataQAA_init_coeff = np.copy(QAA_field.data)
	dataQAB_init_coeff = np.copy(QAB_field.data)
	dataU_init_coeff = np.copy(U_field.data)
	dataV_init_coeff = np.copy(V_field.data)
elif d_input_mode == 'c':
	dataQAA_init_coeff_ = np.array(input_file.get('/QAA_coeff'))
	dataQAB_init_coeff_ = np.array(input_file.get('/QAB_coeff'))
	dataU_init_coeff_ = np.array(input_file.get('/U_coeff'))
	dataV_init_coeff_ = np.array(input_file.get('/V_coeff'))

	dataQAA_init_coeff = np.zeros((NXH, NY), dtype = np.complex128)
	dataQAB_init_coeff = np.zeros((NXH, NY), dtype = np.complex128)
	dataU_init_coeff = np.zeros((NXH, NY), dtype = np.complex128)
	dataV_init_coeff = np.zeros((NXH, NY), dtype = np.complex128)

	# Use 'min(NXH,NXH_input)' to cover both cases when NXH_input > NXH or NXH_input < NXH
	dataQAA_init_coeff[0:min(NXH,NXH_input),0:min(NY,NY_input)] = np.copy(dataQAA_init_coeff_)[0:min(NXH,NXH_input),0:min(NY,NY_input)]
	dataQAB_init_coeff[0:min(NXH,NXH_input),0:min(NY,NY_input)] = np.copy(dataQAB_init_coeff_)[0:min(NXH,NXH_input),0:min(NY,NY_input)]
	dataU_init_coeff[0:min(NXH,NXH_input),0:min(NY,NY_input)] = np.copy(dataU_init_coeff_)[0:min(NXH,NXH_input),0:min(NY,NY_input)]
	dataV_init_coeff[0:min(NXH,NXH_input),0:min(NY,NY_input)] = np.copy(dataV_init_coeff_)[0:min(NXH,NXH_input),0:min(NY,NY_input)]
else:
	if d_input_mode == 'g' and (not NX_input == NX or not NY_input == NY):
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
print('b = ' + str(b))
print('lambda = ' + str(lamb))
print('width = ' + str(width))
print('height = ' + str(height))
print('NX = ' + str(NX))
print('NY = ' + str(NY))
print('initial timestep = ' + str(initial_timestep))
print('---------------------------------')

log_file.writelines('---------------------------------' + '\n')
log_file.writelines('--- Parameters ------------------' + '\n')
log_file.writelines('---------------------------------' + '\n')
log_file.writelines('Ra = ' + str(Ra) + '\n')
log_file.writelines('Er = ' + str(Er) + '\n')
log_file.writelines('Re = ' + str(Re) + '\n')
log_file.writelines('b = ' + str(b) + '\n')
log_file.writelines('lambda = ' + str(lamb) + '\n')
log_file.writelines('width = ' + str(width) + '\n')
log_file.writelines('height = ' + str(height) + '\n')
log_file.writelines('NX = ' + str(NX) + '\n')
log_file.writelines('NY = ' + str(NY) + '\n')
log_file.writelines('initial timestep = ' + str(initial_timestep) + '\n')
log_file.writelines('---------------------------------' + '\n')
log_file.writelines('\n')

# Create bases and domain; to use periodic boundary conditions in the y-direction as well as x, just replace 'Chebyshev' with 'Fourier'
# and delete the Dirichlet boundary conditions  in the 'problem.add_bc' commands. 
x_basis = de.Fourier('x', NX, interval = (-0.5*width, 0.5*width), dealias = 2) # dealias factor is set to 2 to correct for cubic nonlinearities
y_basis = de.Chebyshev('y', NY, interval = (0, height), dealias = 2)
domain = de.Domain([x_basis, y_basis], grid_dtype = np.float64)
xg = domain.grid(0)
yg = domain.grid(1)

#Create problem: define field variables, problem metadata, parameters, equations, and boundary conditions.
#Here 'QAA' is the x,x component of the Q-tensor, and 'QAB' is the x,y component. 
#'U' and 'V' are the x- and y-velocity components.
#'p' is the pressure.
problem = de.IVP(domain, variables=['p','U','V','Ux','Uy','Vx','Vy','QAA','QAAx','QAAy','QAB','QABx','QABy'])
problem.meta[:]['y']['dirichlet'] = True # Tell the problem class that all our boundary conditions are dirichlet. This speeds things up somewhat.

# (constant) parameters
problem.parameters['width'] = width
problem.parameters['height'] = height
problem.parameters['RaEr'] = Ra/Er
problem.parameters['Re'] = Re
problem.parameters['b'] = b # Free energy parameter
problem.parameters['lamb'] = lamb # Flow alignment; note that it is 'lamb' rather than 'lambda' to avoid conflict with the reserved Python keyword of the same name
problem.parameters['Erinv'] = 1/Er

# Equations
problem.add_equation("Ux - dx(U) = 0")
problem.add_equation("Uy - dy(U) = 0")
problem.add_equation("Vx - dx(V) = 0")
problem.add_equation("Vy - dy(V) = 0")
problem.add_equation("QAAx - dx(QAA) = 0")
problem.add_equation("QAAy - dy(QAA) = 0")
problem.add_equation("QABx - dx(QAB) = 0")
problem.add_equation("QABy - dy(QAB) = 0")

problem.add_equation("dt(QAA)-dx(QAAx)-dy(QAAy)-QAA-lamb*Ux=-2*b*QAA*QAA*QAA-2*b*QAA*QAB*QAB-QAAx*U-QAAy*V+QAB*Uy-QAB*Vx")
problem.add_equation("dt(QAB)-dx(QABx)-dy(QABy)-QAB-0.5*lamb*Uy-0.5*lamb*Vx=-2*b*QAA*QAA*QAB-2*b*QAB*QAB*QAB-QAA*Uy+QAA*Vx-QABx*U-QABy*V")
problem.add_equation("Re*dt(U)+dx(p)-2*dx(Ux)-dy(Uy)-dx(Vy)+RaEr*QAAx+RaEr*QABy=-Re*U*Ux-Re*Uy*V")
problem.add_equation("Re*dt(V)+dy(p)-2*dy(Vy)-dx(Uy)-dx(Vx)-RaEr*QAAy+RaEr*QABx=-Re*U*Vx-Re*V*Vy")
problem.add_equation("Ux + Vy = 0")

# Boundary conditions
problem.add_bc("left(U) = 0") # No slip
problem.add_bc("left(V) = 0") #
problem.add_bc("left(QAA) = -0.5") # Strong perpendicular anchoring
problem.add_bc("left(QAB) = 0") #
problem.add_bc("right(U) = 0")
problem.add_bc("right(V) = 0", condition="(nx != 0)")
problem.add_bc("left(p) = 0", condition="(nx == 0)")
problem.add_bc("right(QAA) = -0.5") # Strong perpendicular anchoring
problem.add_bc("right(QAB) = 0") #

# Build and initialize solver (see end of section 4.3)
solver = problem.build_solver(de.timesteppers.RK222)
logger.info('Solver built')
QAA_state_init = solver.state['QAA']
QAB_state_init = solver.state['QAB']
U_state_init = solver.state['U']
V_state_init = solver.state['V']
QAA_state_init['c'] = np.copy(dataQAA_init_coeff)
QAB_state_init['c'] = np.copy(dataQAB_init_coeff)
U_state_init['c'] = np.copy(dataU_init_coeff)
V_state_init['c'] = np.copy(dataV_init_coeff)

# Alternate initialization: specify functional form in grid space (make sure to also comment out the above lines of code initializing in coefficient space)
#QAA_state_init['g'] = -0.5
#QAB_state_init['g'] = 0
#U_state_init['g'] = 0
#V_state_init['g'] = 0

# Output instantaneous channel averages (section 4.6)
order_params = solver.evaluator.add_file_handler('order_params', sim_dt = 0.01, max_size = np.inf)
order_params.add_task("integ(U)/(height*width)", layout='g', name='u_int')
order_params.add_task("integ(V*V)/(height*width)", layout='g', name='v2_int')
order_params.add_task("integ(QAA)/(height*width)", layout='g', name='qaa_int')
order_params.add_task("integ(2*sqrt(QAA*QAA+QAB*QAB))/(height*width)", layout='g', name='S_int') # Channel-averaged order parameter

# Output solution fields (section 4.6)
full_solution = solver.evaluator.add_file_handler('full_solution', sim_dt = 0.04, max_size = np.inf)
#full_solution.add_task('QAA', layout='g', name='QAA')
#full_solution.add_task('QAB', layout='g', name='QAB')
full_solution.add_task('U', layout='g', name='U')
full_solution.add_task('V', layout='g', name='V')
full_solution.add_task('Uy', layout='g', name='Uy')
full_solution.add_task('Vx', layout='g', name='Vx')
full_solution.add_task('2*sqrt(QAA*QAA+QAB*QAB)', layout='g', name='S') # Order parameter field
full_solution.add_task('QAB+1e-7', layout='g', name='nx') # x-component of the director (NOT normalized; this needs to be done in postprocessing)
full_solution.add_task('sqrt(QAA*QAA+QAB*QAB)-QAA', layout='g', name='ny') #y-component of the director
#full_solution.add_task('QAA', layout='c', name='QAA_coeff')
#full_solution.add_task('QAB', layout='c', name='QAB_coeff')
#full_solution.add_task('U', layout='c', name='U_coeff')
#full_solution.add_task('V', layout='c', name='V_coeff')

#Simulation stop conditions (section 4.5)
solver.stop_sim_time = 20
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

# CFL (section 4.4.1 and 4.8)
CFL = flow_tools.CFL(solver, initial_dt = initial_timestep, cadence = 10, max_change = 1.5, safety = 0.5, threshold = 0.1, max_dt = max_timestep)
CFL.add_velocities(('U', 'V'))
dt = CFL.compute_dt()

# Main loop
while solver.ok:
	solver.step(dt) # Iterate solver once with timestep dt
	if solver.iteration % 100 == 0: # Log output every 1000 iterations
		logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
		log_file.writelines('Iteration = ' + str(solver.iteration) + ';   Time = ' + str(solver.sim_time) + ';   dt = ' + str(dt) + '\n')
	dt = CFL.compute_dt() #Update the timestep via CFL
	
# Example of manually outputting solution data to an HDF5 file. Here, the x-derivative of U 
# in grid space is written to 'Ux.h5'. 
Ux_state_final = solver.state['Ux']
Ux_state_final.require_grid_space()
h5f = h5py.File('Ux.h5', 'w')
h5f.create_dataset('Ux', data = Ux_state_final.data)
h5f.close()
	
stop = timeit.default_timer()
print('Time: ' + str(stop-start) + ' seconds')
log_file.writelines('Time: ' + str(stop-start) + ' seconds')
log_file.close()

















