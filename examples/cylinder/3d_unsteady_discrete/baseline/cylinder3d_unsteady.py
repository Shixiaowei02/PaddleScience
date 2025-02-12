# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# discrete time method

import paddlescience as psci
import numpy as np
import paddle
import os
import wget
import zipfile

paddle.seed(1)
np.random.seed(1)

#paddle.enable_static()
paddle.disable_static()


# load real data 
def GetRealPhyInfo(time, need_info=None):
    # if real data don't exist, you need to download it.
    if os.path.exists('./openfoam_cylinder_re100') == False:
        data_set = 'https://dataset.bj.bcebos.com/PaddleScience/cylinder3D/openfoam_cylinder_re100/cylinder3d_openfoam_re100.zip'
        wget.download(data_set)
        with zipfile.ZipFile('cylinder3d_openfoam_re100.zip', 'r') as zip_ref:
            zip_ref.extractall('openfoam_cylinder_re100')
    real_data = np.load("openfoam_cylinder_re100/flow_re100_" + str(
        int(time)) + "_xyzuvwp.npy")
    real_data = real_data.astype(np.float32)
    if need_info == 'cord':
        return real_data[:, 0:3]
    elif need_info == 'physic':
        return real_data[:, 3:7]
    else:
        return real_data


# define start time
start_time = 100

cc = (0.0, 0.0)
cr = 0.5
geo = psci.geometry.CylinderInCube(
    origin=(-8, -8, -2), extent=(25, 8, 2), circle_center=cc, circle_radius=cr)

geo.add_boundary(name="left", criteria=lambda x, y, z: abs(x + 8.0) < 1e-4)
geo.add_boundary(name="right", criteria=lambda x, y, z: abs(x - 25.0) < 1e-4)
geo.add_boundary(
    name="circle",
    criteria=lambda x, y, z: ((x - cc[0])**2 + (y - cc[1])**2 - cr**2) < 1e-4)

# discretize geometry
geo_disc = geo.discretize(npoints=[200, 50, 4], method="uniform")

# the real_cord need to be added in geo_disc
geo_disc.user = GetRealPhyInfo(start_time, need_info='cord')

# N-S equation
pde = psci.pde.NavierStokes(
    nu=0.01,
    rho=1.0,
    dim=3,
    time_dependent=True,
    weight=[0.01, 0.01, 0.01, 0.01])

pde.set_time_interval([100.0, 110.0])

# boundary condition on left side: u=1, v=w=0
bc_left_u = psci.bc.Dirichlet('u', rhs=1.0, weight=1.0)
bc_left_v = psci.bc.Dirichlet('v', rhs=0.0, weight=1.0)
bc_left_w = psci.bc.Dirichlet('w', rhs=0.0, weight=1.0)

# boundary condition on right side: p=0
bc_right_p = psci.bc.Dirichlet('p', rhs=0.0, weight=1.0)

# boundary on circle
bc_circle_u = psci.bc.Dirichlet('u', rhs=0.0, weight=1.0)
bc_circle_v = psci.bc.Dirichlet('v', rhs=0.0, weight=1.0)
bc_circle_w = psci.bc.Dirichlet('w', rhs=0.0, weight=1.0)

# add bounday and boundary condition
pde.add_bc("left", bc_left_u, bc_left_v, bc_left_w)
pde.add_bc("right", bc_right_p)
pde.add_bc("circle", bc_circle_u, bc_circle_v, bc_circle_w)

# pde discretization 
pde_disc = pde.discretize(
    time_method="implicit", time_step=1, geo_disc=geo_disc)

# Network
net = psci.network.FCNet(
    num_ins=3, num_outs=4, num_layers=10, hidden_size=50, activation='tanh')

# Loss
loss = psci.loss.L2(p=2, data_weight=100.0)

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

# Solver parameter
solver = psci.solver.Solver(pde=pde_disc, algo=algo, opt=opt)

# train
# Solver time: (100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
current_interior = np.zeros(
    (len(pde_disc.geometry.interior), 3)).astype(np.float32)
current_user = GetRealPhyInfo(start_time, need_info='physic')[:, 0:3]
for next_time in range(
        int(pde_disc.time_internal[0]) + 1,
        int(pde_disc.time_internal[1]) + 1):
    print("### train next time=%f train task ###" % next_time)
    solver.feed_data_interior_cur(current_interior)  # add u(n) interior
    solver.feed_data_user_cur(current_user)  # add u(n) user 
    solver.feed_data_user_next(GetRealPhyInfo(
        next_time, need_info='physic'))  # add u(n+1) user
    next_uvwp = solver.solve(
        num_epoch=2000,
        checkpoint_path='checkpoint/cylinder3d_model_' + str(next_time) + "/")
    # Save vtk
    file_path = "train_cylinder_unsteady_re100/cylinder3d_train_rslt_" + str(
        next_time)
    psci.visu.save_vtk(
        filename=file_path, geo_disc=pde_disc.geometry, data=next_uvwp)
    # current_info need to be modified as follows: current_time -> next time
    current_interior = np.array(next_uvwp[0])[:, 0:3]
    current_user = np.array(next_uvwp[-1])[:, 0:3]

# predict
cc = (0.0, 0.0)
cr = 1.0
geo_predict = psci.geometry.CylinderInCube(
    origin=(-8, -8, -2), extent=(8, 8, 2), circle_center=cc, circle_radius=cr)

solver.pde.geometry = geo_predict.discretize(npoints=5000, method="sampling")

if paddle.in_dynamic_mode():
    next_uvwp = solver.predict(
        dynamic_net_file='checkpoint/cylinder3d_model_101/dynamic_net_params_1000.pdparams',
        dynamic_opt_file='checkpoint/cylinder3d_model_101/dynamic_opt_params_1000.pdopt'
    )
else:
    next_uvwp = solver.predict(
        static_model_file='checkpoint/cylinder3d_model_101/static_model_params_1000.pdparams'
    )

# save vtk
if paddle.in_dynamic_mode():
    file_path = "predict_cylinder_unsteady_re100/rslt_dynamic_" + str(100)
else:
    file_path = "predict_cylinder_unsteady_re100/rslt_static_" + str(100)
psci.visu.save_vtk(
    filename=file_path, geo_disc=pde_disc.geometry, data=next_uvwp)

# save npy
result = next_uvwp[0]
result = np.array(result)
np.save("predict_cylinder_unsteady_re100/predict_result.npy", result)
