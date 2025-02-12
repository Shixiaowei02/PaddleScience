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


def compare(res, expect, delta=1e-6, rtol=1e-5):
    """
    比较函数
    :param paddle: paddle结果
    :param torch: torch结果
    :param delta: 误差值
    :return:
    """
    if isinstance(res, np.ndarray):
        print(1)
        assert res.shape == expect.shape
        assert np.allclose(res, expect, atol=delta, rtol=rtol, equal_nan=True)
    elif isinstance(res, (list, tuple)):
        for i, j in enumerate(res):
            compare(j, expect[i], delta, rtol)
            print(2)
    elif isinstance(res, (int, float, complex, bool)):
        assert np.allclose(res, expect, atol=delta, rtol=rtol, equal_nan=True)
        print(3)
    else:
        assert TypeError


if __name__ == '__main__':
    a = 1
    b = 1
    compare(a, b)
