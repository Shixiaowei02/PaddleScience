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

import numpy as np
import copy
import types
import vtk
from pyevtk.hl import pointsToVTK
import paddle
from .. import config


# Save geometry pointwise
def save_vtk(filename="output", time_array=None, geo_disc=None, data=None):
    """
    Visualization. Save data to vtk format.

    Parameters:
        filename(string): file name.
        time_array(list or numpy array, optional): time steps list / array.
        geo_disc (GeometryDiscrete): discrete geometry.
        data (numpy array): data to be visualized.


    Example:
        >>> import paddlescience as psci
        >>> psci.visu.save_vtk(geo_disc=pde_disc.geometry, data=solution)
    """

    nt = 1 if (time_array is None) else len(time_array) - 1
    nprocs = paddle.distributed.get_world_size()
    nrank = paddle.distributed.get_rank()
    if nprocs == 1:
        geo_disc_sub = geo_disc
    else:
        geo_disc_sub = geo_disc.sub(nprocs, nrank)

    # concatenate data and cordiante 
    points_vtk = __concatenate_geo(geo_disc_sub)

    # points's shape is [ndims][npoints]
    npoints = len(points_vtk[0])
    ndims = len(points_vtk)

    # data
    if data is None:
        data_vtk = {"placeholder": np.ones(npoints, dtype=config._dtype)}
    elif type(data) == types.LambdaType:
        data_vtk = dict()
        if ndims == 3:
            data_vtk["data"] = data(points_vtk[0], points_vtk[1],
                                    points_vtk[2])
        elif ndims == 2:
            data_vtk["data"] = data(points_vtk[0], points_vtk[1])
    else:
        data_vtk = __concatenate_data(data, nt)

    if ndims == 3:
        axis_x = points_vtk[0]
        axis_y = points_vtk[1]
        axis_z = points_vtk[2]
        for t in range(nt):
            fpname = filename + "-t" + str(t + 1) + "-p" + str(nrank)
            pointsToVTK(fpname, axis_x, axis_y, axis_z, data=data_vtk[t])
    elif ndims == 2:
        axis_x = points_vtk[0]
        axis_y = points_vtk[1]
        axis_z = np.zeros(npoints, dtype=config._dtype)
        for t in range(nt):
            fpname = filename + "-t" + str(t + 1) + "-p" + str(nrank)
            pointsToVTK(fpname, axis_x, axis_y, axis_z, data=data_vtk[t])


def __save_vtk_raw(filename="output", cordinate=None, data=None):

    npoints = len(cordinate)
    ndims = len(cordinate[0])

    if data is None:
        data = np.ones((npoints, 1), dtype=type(cordinate[0, 0]))

    data_vtk = dict()

    for i in range(len(data[0, :])):
        data_vtk[str(i + 1)] = data[:, i].copy()

    if ndims == 3:
        axis_x = cordinate[:, 0].copy()
        axis_y = cordinate[:, 1].copy()
        axis_z = cordinate[:, 2].copy()
        pointsToVTK(filename, axis_x, axis_y, axis_z, data=data_vtk)
    elif ndims == 2:
        axis_x = cordinate[:, 0].copy()
        axis_y = cordinate[:, 1].copy()
        axis_z = np.zeros(npoints, dtype=config._dtype)
        pointsToVTK(filename, axis_x, axis_y, axis_z, data=data_vtk)


# concatenate cordinates of interior points and boundary points
def __concatenate_geo(geo_disc):

    # concatenate interior and bounday points
    x = [geo_disc.interior]
    for value in geo_disc.boundary.values():
        x.append(value)
    points = np.concatenate(x, axis=0)

    ndims = len(points[0])

    # to pointsToVTK input format
    points_vtk = list()
    for i in range(ndims):
        points_vtk.append(points[:, i].copy())

    return points_vtk


# concatenate data
def __concatenate_data(outs, nt=1):

    vtkname = ["u1", "u2", "u3", "u4", "u5"]

    data = dict()

    # to numpy
    npouts = list()
    for out in outs:
        if type(out) != np.ndarray:
            npouts.append(out.numpy())  # tenor to array
        else:
            npouts.append(out)

    # concatenate data
    ndata = outs[0].shape[1]
    data_vtk = list()

    for t in range(nt):
        for i in range(ndata):
            x = list()
            for out in npouts:
                s = int(len(out) / nt) * t
                e = int(len(out) / nt) * (t + 1)
                x.append(out[s:e, i])
            data[vtkname[i]] = np.concatenate(x, axis=0)

        data_vtk.append(data)

    return data_vtk
