# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Bidirectional LM encoder."""
__all__ = ['BiLMEncoder']

import mxnet as mx
from mxnet import gluon
from .utils import _get_rnn_cell_clip_residual

class BiLMEncoder(gluon.Block):
    def __init__(self, mode, num_layers, input_size, hidden_size, dropout, skip_connection, proj_size=None, cell_clip=None, proj_clip=None, bidirectional=True):
        super(BiLMEncoder, self).__init__()

        self.num_layers = num_layers

        lstm_input_size = input_size

        with self.name_scope():
            for layer_index in range(num_layers):
                forward_layer = _get_rnn_cell_clip_residual(mode=mode, num_layers=1, input_size=lstm_input_size, hidden_size=hidden_size,
                                              dropout=0 if layer_index == num_layers - 1 else dropout, skip_connection=False if layer_index == 0 else skip_connection,
                                              proj_size=proj_size, cell_clip=cell_clip, proj_clip=proj_clip)
                backward_layer = _get_rnn_cell_clip_residual(mode=mode, num_layers=1, input_size=lstm_input_size, hidden_size=hidden_size,
                                              dropout=0 if layer_index == num_layers - 1 else dropout, skip_connection=False if layer_index == 0 else skip_connection,
                                              proj_size=proj_size, cell_clip=cell_clip, proj_clip=proj_clip)

                setattr(self, 'forward_layer_{}'.format(layer_index), forward_layer)
                setattr(self, 'backward_layer_{}'.format(layer_index), backward_layer)

                lstm_input_size = proj_size if mode == 'lstmp' else hidden_size

    def begin_state(self, *args, **kwargs):
        return [getattr(self, 'forward_layer_{}'.format(layer_index)).begin_state(*args, **kwargs) for layer_index in range(self.num_layers)],\
               [getattr(self, 'backward_layer_{}'.format(layer_index)).begin_state(*args, **kwargs) for layer_index in range(self.num_layers)]

    def forward(self, inputs, states):
        seq_len = inputs[0].shape[0]

        if not states:
            states_forward, states_backward = self.begin_state(batch_size=inputs[0].shape[1])
        else:
            states_forward, states_backward = states

        outputs_forward = []
        outputs_backward = []

        for j in range(self.num_layers):
            outputs_forward.append([])
            for i in range(seq_len):
                if j == 0:
                    output, states_forward[j] = getattr(self, 'forward_layer_{}'.format(j))(inputs[0][i], states_forward[j])
                else:
                    output, states_forward[j] = getattr(self, 'forward_layer_{}'.format(j))(outputs_forward[j-1][i], states_forward[j])
                outputs_forward[j].append(output)

            outputs_backward.append([None] * seq_len)
            for i in reversed(range(seq_len)):
                if j == 0:
                    output, states_backward[j] = getattr(self, 'backward_layer_{}'.format(j))(inputs[1][i], states_backward[j])
                else:
                    output, states_backward[j] = getattr(self, 'backward_layer_{}'.format(j))(outputs_backward[j-1][i], states_backward[j])
                outputs_backward[j][i] = output

        for i in range(self.num_layers):
            outputs_forward[i] = mx.nd.stack(*outputs_forward[i])
            outputs_backward[i] = mx.nd.stack(*outputs_backward[i])

        return (outputs_forward, outputs_backward), (states_forward, states_backward)
