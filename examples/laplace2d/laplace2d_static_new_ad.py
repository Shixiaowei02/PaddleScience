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

import paddle
import time
import six
import os
import paddlescience as psci
import paddle.compat as cpt
import numpy as np
from paddle import fluid
from paddle.fluid import core
from paddle.fluid.framework import Variable
from paddle.fluid.incubate.ad_transform.primx import enable_prim, prim2orig, prim_enabled
from paddle.static import global_scope

paddle.enable_static()
paddle.seed(1234)
np.random.seed(1234)
enable_prim()

convert_back_to_program = (os.getenv('FLAGS_use_cinn') != "1")


def cinn_optimize_program(program):
    def _remove_unused_var(program):
        all_remove_vars = []
        for block in program.blocks:
            args = []
            for op in block.ops:
                args += op.input_arg_names
                args += op.output_arg_names
            args = list(set(args))  # vals of all left ops
            var_names = block.vars.keys()  # all vals
            sub_block_remove_vars = []
            for var in var_names:
                if var not in args:
                    sub_block_remove_vars.append(var)
            all_remove_vars.append(sub_block_remove_vars)

        remove_vars = [list(set(v)) for v in all_remove_vars]
        for i, block in enumerate(program.blocks):
            for v in remove_vars[i]:
                block._remove_var(v)

    def dead_code_elimination(program):
        program._sync_with_cpp()
        all_input_arg_names = set()
        for block in program.blocks:
            ops = list(block.ops)
            for op in ops:
                for name in op.input_arg_names:
                    all_input_arg_names.add(name)

        for block in program.blocks:
            ops = list(block.ops)
            for op in ops:
                if op.type == "fill_constant_p" and (
                        op.output('Y')[0] not in all_input_arg_names):
                    idx = block.ops.index(op)
                    block._remove_op(idx)

        _remove_unused_var(program)
        program._sync_with_cpp()

    def fuse_shape_fill_constant(program):
        def _insert_fill_any_like_op(block, index, shape_op, fill_constant_op):
            fill_any_like_inputs = {}
            fill_any_like_inputs['X'] = block.var(shape_op.input('Input')[0])
            fill_any_like_outputs = {}
            fill_any_like_outputs['Out'] = block.var(
                fill_constant_op.output('Out')[0])
            fill_any_like_attrs = {}
            fill_any_like_attrs['value'] = fill_constant_op.attr('value')
            fill_any_like_attrs['dtype'] = fill_constant_op.attr('dtype')
            fill_any_like_attrs['op_role'] = fill_constant_op.attr('op_role')

            fill_any_like_op = block._insert_op(
                index,
                type='fill_any_like',
                inputs=fill_any_like_inputs,
                outputs=fill_any_like_outputs,
                attrs=fill_any_like_attrs)
            return fill_any_like_op

        program._sync_with_cpp()
        block = program.block(0)
        i = 0
        while i < len(block.ops):
            # find a fill_constant op
            if block.ops[i].type == 'fill_constant':
                fill_constant_op = block.ops[i]
                fill_constant_idx = i
                shape_idx = -1
                # find the preceding shape op
                for j in reversed(range(fill_constant_idx)):
                    if block.ops[j].type == 'shape':
                        shape_out_name = block.ops[j].output_arg_names[0]
                        if shape_out_name in fill_constant_op.input_arg_names:
                            shape_op = block.ops[j]
                            shape_idx = j
                            break
                if shape_idx < 0:
                    i += 1
                    continue
                # create and insert a new fill_any_like op
                _insert_fill_any_like_op(block, fill_constant_idx + 1,
                                         shape_op, fill_constant_op)
                # remove the old operators
                block._remove_op(fill_constant_idx)
                block._remove_op(shape_idx)
                # restart scanning for elementwise add from the deleted shape's index
                i = shape_idx
            i += 1
        _remove_unused_var(program)
        program._sync_with_cpp()

    dead_code_elimination(train_program)
    fuse_shape_fill_constant(train_program)


def add_fetch_ops(program, fetch_list, fetch_var_name='fetch'):
    assert isinstance(program, fluid.Program)
    global_block = program.global_block()

    if fetch_var_name in global_block.vars:
        fetch_var = global_block.var(fetch_var_name)
    else:
        fetch_var = global_block.create_var(
            name=fetch_var_name,
            type=core.VarDesc.VarType.FETCH_LIST,
            persistable=True)

    # append fetch_operators
    if not fluid.executor.has_fetch_operators(global_block, fetch_list,
                                              fetch_var_name, 'fetch'):
        for i, var in enumerate(fetch_list):
            assert isinstance(var, Variable) or isinstance(
                var, six.string_types), ("Wrong type for fetch_list[%s]: %s" %
                                         (i, type(var)))
            global_block.append_op(
                type='fetch',
                inputs={'X': [var]},
                outputs={'Out': [fetch_var]},
                attrs={'col': i})
    program._sync_with_cpp()


def compile(program, loss_name=None):
    build_strategy = paddle.static.BuildStrategy()
    exec_strategy = paddle.static.ExecutionStrategy()

    exec_strategy.num_threads = 1

    compiled_program = paddle.static.CompiledProgram(
        program).with_data_parallel(
            loss_name=loss_name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)

    return compiled_program


def compile_and_convert_back_to_program(program=None,
                                        feed=None,
                                        fetch_list=None,
                                        fetch_var_name='fetch',
                                        scope=None,
                                        use_prune=False,
                                        loss_name=None):
    def _add_fetch_ops(program, fetch_list, fetch_var_name):
        assert isinstance(program, fluid.Program)
        tmp_program = program.clone()
        global_block = tmp_program.global_block()

        if fetch_var_name in global_block.vars:
            fetch_var = global_block.var(fetch_var_name)
        else:
            fetch_var = global_block.create_var(
                name=fetch_var_name,
                type=core.VarDesc.VarType.FETCH_LIST,
                persistable=True)

        # append fetch_operators
        if not fluid.executor.has_fetch_operators(global_block, fetch_list,
                                                  fetch_var_name, 'fetch'):
            for i, var in enumerate(fetch_list):
                assert isinstance(var, Variable) or isinstance(
                    var, six.string_types), (
                        "Wrong type for fetch_list[%s]: %s" % (i, type(var)))
                global_block.append_op(
                    type='fetch',
                    inputs={'X': [var]},
                    outputs={'Out': [fetch_var]},
                    attrs={'col': i})
        return tmp_program

    def _remove_fetch_ops(program):
        assert isinstance(program, fluid.Program)
        tmp_program = program.clone()
        global_block = tmp_program.global_block()
        op_num = len(global_block.ops)
        for idx in reversed(range(op_num)):
            if global_block.ops[idx].type == 'fetch':
                global_block._remove_op(idx)

        return tmp_program

    if program is None:
        program = default_main_program()

    if scope is None:
        scope = global_scope()

    executor = paddle.static.Executor()

    fetch_list = executor._check_fetch_list(fetch_list)
    fetch_list, optimize_ops = executor._split_optimize_ops_in_fetch_list(
        fetch_list)

    if optimize_ops:
        raise ValueError("Unsupport to fetch optimize OP.")

    if use_prune:
        program = executor._prune_program(program, feed, fetch_list,
                                          optimize_ops)
        feed = executor._update_feed(program, feed)

    program_with_fetch_op = _add_fetch_ops(program, fetch_list, fetch_var_name)
    compiled_program = compile(program_with_fetch_op, loss_name)
    assert isinstance(compiled_program, fluid.compiler.CompiledProgram)

    compiled_program._compile(scope,
                              paddle.framework._current_expected_place())
    compiled_graph = compiled_program._graph
    ir_graph = fluid.framework.IrGraph(compiled_graph, for_test=True)
    #ir_graph.draw(save_path='./', name='compiled_graph')
    ir_program = ir_graph.to_program()
    final_program = _remove_fetch_ops(ir_program)

    #paddle.static.save(final_program, "final")
    return final_program


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
pdes, geo = psci.discretize(pdes, geo, space_nsteps=(101, 101))

# bc value
golden, bc_value = GenSolution(geo.get_space_domain(), geo.get_bc_index())
pdes.set_bc_value(bc_value=bc_value)

# psci.visu.save_vtk(geo, golden, 'golden_laplace_2d')
# np.save('./golden_laplace_2d.npy', golden)

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
        num_layers=10,
        hidden_size=50,
        dtype='float32',
        activation='tanh')

    outputs = net.nn_func(inputs)

    # bc_loss
    bc_index = paddle.static.data(name='bc_idx', shape=[400], dtype='int32')
    bc_value = paddle.static.data(name='bc_v', shape=[400, 1], dtype='float32')
    bc_u = paddle.index_select(outputs, bc_index)
    bc_diff = bc_u - bc_value
    bc_loss = paddle.norm(bc_diff, p=2)

    # eq_loss
    jac, = paddle.static.gradients([outputs], [inputs])
    hes_0, = paddle.static.gradients([jac[:, 0]], [inputs])
    hes_1, = paddle.static.gradients([jac[:, 1]], [inputs])
    eq_loss = paddle.norm(hes_0[:, 0] + hes_1[:, 1], p=2)

    loss = bc_loss + eq_loss
    paddle.fluid.optimizer.AdamOptimizer(0.001).minimize(loss)
    if convert_back_to_program and prim_enabled():
        prim2orig(inputs.block)

# print('startup_program: ', startup_program)
# print('train_program: ', train_program)

exe.run(startup_program)
num_epoch = 2010

feeds = {
    'x': geo.get_space_domain().astype(np.float32),
    'bc_idx': geo.bc_index.astype(np.int32),
    'bc_v': pdes.bc_value
}
fetchs = [loss.name, eq_loss.name, bc_loss.name, outputs.name]

if convert_back_to_program:
    print("Run without CINN")
    compiled_program = compile_and_convert_back_to_program(
        train_program,
        feed=feeds,
        fetch_list=fetchs,
        use_prune=True,
        loss_name=loss.name)
else:
    print("Run with CINN")
    cinn_optimize_program(train_program)
    add_fetch_ops(train_program, fetch_list=fetchs)
    compiled_program = compile(train_program, loss.name)

print("Get train program successfully, congratulations !!!")

begin = time.time()
for i in range(num_epoch):
    if i == 10:
        paddle.device.cuda.synchronize()
        begin = time.time()
        print("begin With CINN at ", begin)

    loss_d, eq_loss_d, bc_loss_d, outputs_d = exe.run(compiled_program,
                                                      feed=feeds,
                                                      fetch_list=fetchs)
    print('num_epoch: ', i, '/', num_epoch, ' loss: ', loss_d[0], ' eq_loss: ',
          eq_loss_d[0], 'bc_loss: ', bc_loss_d[0], 'outputs[0][0]: ',
          outputs_d[0][0])

paddle.device.cuda.synchronize()
end = time.time()
print('{} epoch(10~{}) time: {} s'.format(num_epoch - 10, num_epoch, end -
                                          begin))

rslt = exe.run(compiled_program,
               feed={
                   'x': geo.get_space_domain().astype(np.float32),
                   'bc_idx': geo.bc_index.astype(np.int32),
                   'bc_v': pdes.bc_value
               },
               fetch_list=[outputs.name, ])[0]
# psci.visu.save_vtk(geo, rslt, 'rslt_laplace_2d')
# np.save('./rslt_laplace_2d.npy', rslt)

# Calculate diff and l2 relative error
diff = rslt - golden
# psci.visu.save_vtk(geo, diff, 'diff_laplace_2d')
# np.save('./diff_laplace_2d.npy', diff)
root_square_error = np.linalg.norm(diff, ord=2)
mean_square_error = root_square_error * root_square_error / geo.get_domain_size(
)
print('mean_sqeare_error: ', mean_square_error)
