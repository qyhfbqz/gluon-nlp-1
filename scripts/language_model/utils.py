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
"""Language model utilities."""
from mxnet import nd, autograd

def detach(hidden):
    """Transfer hidden states into new states, to detach them from the history.
    Parameters
    ----------
    hidden : NDArray
        The hidden states

    Returns
    ----------
    hidden: NDArray
        The detached hidden states
    """
    if isinstance(hidden, (tuple, list)):
        hidden = [detach(h) for h in hidden]
    else:
        hidden = hidden.detach()
    return hidden

def forward(model, inputs, begin_state=None):
    """Implement the forward computation that the awd language model and cache model use.

    Parameters
    ----------
    inputs : NDArray
        The training dataset.
    begin_state : list
        The initial hidden states.

    Returns
    -------
    out: NDArray
        The output of the model.
    out_states: list
        The list of output states of the model's encoder.
    encoded_raw: list
        The list of outputs of the model's encoder.
    encoded_dropped: list
        The list of outputs with dropout of the model's encoder.
    """
    encoded = model.embedding(inputs)
    if not begin_state:
        begin_state = model.begin_state(batch_size=inputs.shape[1])
    out_states = []
    encoded_raw = []
    encoded_dropped = []
    if 'awd' in model.prefix:
        for i, (e, s) in enumerate(zip(model.encoder, begin_state)):
            encoded, state = e(encoded, s)
            encoded_raw.append(encoded)
            out_states.append(state)
            if model._drop_h and i != len(model.encoder) - 1:
                encoded = nd.Dropout(encoded, p=model._drop_h, axes=(0,))
                encoded_dropped.append(encoded)
    else:
        encoded, state = model.encoder(encoded, begin_state)
        encoded_raw.append(encoded)
    if model._dropout:
        encoded = nd.Dropout(encoded, p=model._dropout, axes=(0,))
    if 'awd' in model.prefix:
        encoded_dropped.append(encoded)
        with autograd.predict_mode():
            out = model.decoder(encoded)
    else:
        out = model.decoder(encoded)
    if 'awd' in model.prefix:
        return out, out_states, encoded_raw, encoded_dropped
    else:
        return out, state, encoded_raw, encoded_dropped
