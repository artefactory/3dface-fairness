"""
This file is part of the repo: https://github.com/czh-98/REALY
If you find the code useful, please cite our paper:
"REALY: Rethinking the Evaluation of 3D Face Reconstruction"
European Conference on Computer Vision 2022
Code: https://github.com/czh-98/REALY
Copyright (c) [2021-2022] [Tencent AI Lab]
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np


def fit_icp_RT(source, target, with_scale=True):
    """

    Args:
        source: float vertices, shape: n1x3
        target: fixed vertices, shape: n1x3
        with_scale: whether use the scale factor, bool

    Returns: transformation matrix, scale, rotation matrix, translation matrix

    """

    assert source.shape[0] == 3

    npoint = source.shape[1]
    means = np.mean(source, 1)
    meant = np.mean(target, 1)
    s1 = source - np.tile(means, (npoint, 1)).transpose()
    t1 = target - np.tile(meant, (npoint, 1)).transpose()
    W = t1.dot(s1.transpose())
    U, sig, V = np.linalg.svd(W)
    rotation = U.dot(V)

    scale = np.sum(np.sum(abs(t1))) / np.sum(np.sum(abs(rotation.dot(s1)))) if with_scale else 1.0

    translation = target - scale * rotation.dot(source)
    translation = np.mean(translation, 1)

    trans = np.zeros((4, 4))
    trans[3, 3] = 1
    trans[:3, 0:3] = scale * rotation[:, 0:3]
    trans[:3, 3] = translation[:]

    return trans, scale, rotation, translation
