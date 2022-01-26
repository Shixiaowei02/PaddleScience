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

import paddlescience as psci
import numpy as np

import paddle

paddle.enable_static()
paddle.seed(1234)


# Analytical solution
def LaplaceRecSolution(x, y, k=1.0):
    if (k == 0.0):
        return x * y
    else:
        return np.cos(k * x) * np.cosh(k * y)


# Generate analytical Solution using Geometry points
def GenSolution(xy, bc_index):
    sol = np.zeros((len(xy), 1)).astype(np.float32)
    bc_value = np.zeros((len(bc_index), 1)).astype(np.float32)
    for i in range(len(xy)):
        sol[i] = LaplaceRecSolution(xy[i][0], xy[i][1])
    for i in range(len(bc_index)):
        bc_value[i][0] = sol[bc_index[i]]
    return [sol, bc_value]


# Geometry
geo = psci.geometry.Rectangular(
    space_origin=(0.0, 0.0), space_extent=(1.0, 1.0))

# PDE Laplace
pdes = psci.pde.Laplace2D()

# Discretization
pdes, geo = psci.discretize(pdes, geo, space_nsteps=(11, 11))

# bc value
golden, bc_value = GenSolution(geo.get_space_domain(), geo.get_bc_index())
pdes.set_bc_value(bc_value=bc_value)

psci.visu.save_vtk(geo, golden, 'golden_laplace_2d')
np.save("./golden_laplace_2d.npy", golden)

place = paddle.CUDAPlace(0)
exe = paddle.static.Executor(place)

train_program = paddle.static.Program()
startup_program = paddle.static.Program()
with paddle.static.program_guard(train_program, startup_program):
    inputs = paddle.static.data(
        name='x', shape=[geo.get_domain_size(), 2], dtype='float32')
    inputs.stop_gradient = False
    # Network
    net = psci.network.FCNetStatic(
        num_ins=2,
        num_outs=1,
        num_layers=5,
        hidden_size=20,
        dtype="float32",
        activation="tanh")

    outputs = net.nn_func(inputs)

    # eq_loss
    du = paddle.static.gradients(outputs, inputs)[0]
    d2u_dx2 = paddle.static.gradients(du[:, 0], inputs)[0][:, 0]
    d2u_dy2 = paddle.static.gradients(du[:, 1], inputs)[0][:, 1]
    eq_loss = paddle.norm(d2u_dx2 + d2u_dy2, p=2)
    # bc_loss
    bc_index = paddle.static.data(name='bc_idx', shape=[40], dtype='int64')
    bc_value = paddle.static.data(name='bc_v', shape=[40, 1], dtype='float32')
    bc_u = paddle.index_select(outputs, bc_index)
    bc_diff = bc_u - bc_value
    bc_loss = paddle.norm(bc_diff, p=2)
    loss = eq_loss + bc_loss
    paddle.optimizer.Adam(learning_rate=0.001).minimize(loss)

# print("startup_program: ", startup_program)
# print("train_program: ", train_program)

exe.run(startup_program)
num_epoch = 10
print()

for i in range(num_epoch):
    loss_d, eq_loss_d, bc_loss_d = exe.run(
        train_program,
        feed={
            "x": geo.get_space_domain().astype(np.float32),
            "bc_idx": geo.bc_index,
            "bc_v": pdes.bc_value
        },
        fetch_list=[loss.name, eq_loss.name, bc_loss.name])
    print("num_epoch: ", i, "/", num_epoch, " loss: ", loss_d[0], " eq_loss: ",
          eq_loss_d[0], " bc_loss: ", bc_loss_d[0])
