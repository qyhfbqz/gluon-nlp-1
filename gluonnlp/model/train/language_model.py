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
"""Language models."""
__all__ = ['BiRNN']

from mxnet import init, nd, autograd
from mxnet.gluon import nn, Block

from gluonnlp.model.bilm_encoder import BiLMEncoder

class BiRNN(Block):
    """Standard RNN language model.

    Parameters
    ----------
    mode : str
        The type of RNN to use. Options are 'lstm', 'gru', 'rnn_tanh', 'rnn_relu'.
    vocab_size : int
        Size of the input vocabulary.
    embed_size : int
        Dimension of embedding vectors.
    hidden_size : int
        Number of hidden units for RNN.
    num_layers : int
        Number of RNN layers.
    dropout : float
        Dropout rate to use for encoder output.
    tie_weights : bool, default False
        Whether to tie the weight matrices of output dense layer and input embedding layer.
    """
    def __init__(self, mode, vocab_size, embed_size, hidden_size, num_layers, tie_weights=False, dropout=0.5,
                 skip_connection=False, proj_size=None, proj_clip=None, cell_clip=None, **kwargs):
        if tie_weights:
            assert embed_size == hidden_size, 'Embedding dimension must be equal to ' \
                                              'hidden dimension in order to tie weights. ' \
                                              'Got: emb: {}, hid: {}.'.format(embed_size,
                                                                              hidden_size)
        super(BiRNN, self).__init__(**kwargs)
        self._mode = mode
        self._embed_size = embed_size
        self._hidden_size = hidden_size
        self._skip_connection = skip_connection
        self._proj_size = proj_size
        self._proj_clip = proj_clip
        self._cell_clip = cell_clip
        self._num_layers = num_layers
        self._dropout = dropout
        self._tie_weights = tie_weights
        self._vocab_size = vocab_size

        with self.name_scope():
            self.embedding = self._get_embedding()
            self.encoder = self._get_encoder()
            self.decoder = self._get_decoder()

    def _get_embedding(self):
        embedding = nn.HybridSequential()
        with embedding.name_scope():
            embedding.add(nn.Embedding(self._vocab_size, self._embed_size,
                                       weight_initializer=init.Uniform(0.1), sparse_grad=True))## TODO check sparse_grad
            if self._dropout:
                embedding.add(nn.Dropout(self._dropout))
        return embedding

    def _get_encoder(self):
        return BiLMEncoder(mode=self._mode, num_layers=self._num_layers, input_size=self._embed_size,
                           hidden_size=self._hidden_size, dropout=self._dropout, skip_connection=self._skip_connection,
                           proj_size=self._proj_size, cell_clip=self._cell_clip, proj_clip=self._proj_clip)

    def _get_decoder(self):
        output = nn.HybridSequential()
        with output.name_scope():
            if self._tie_weights:
                output.add(nn.Dense(self._vocab_size, flatten=False,
                                    params=self.embedding[0].params))
            else:
                output.add(nn.Dense(self._vocab_size, flatten=False))
        return output

    def begin_state(self, *args, **kwargs):
        return self.encoder.begin_state(*args, **kwargs)

    def forward(self, inputs, begin_state=None): # pylint: disable=arguments-differ
        """Defines the forward computation. Arguments can be either
        :py:class:`NDArray` or :py:class:`Symbol`."""
        """Implement forward computation.

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
        ## TODO: the embedding and decoder of the forward and backward should be tied
        encoded = self.embedding(inputs[0]), self.embedding(inputs[1])

        if not begin_state:
            ## TODO: check shape
            begin_state = self.begin_state(inputs[0].shape[1])
        ## TODO: check state output
        out_states = []
        encoded_raw = []
        encoded_dropped = []

        encoded, state = self.encoder(encoded, begin_state)
        encoded_raw.append(encoded)

        if self._dropout:
            encoded_forward = nd.Dropout(encoded[0][-1], p=self._dropout)
            encoded_backward = nd.Dropout(encoded[1][-1], p=self._dropout)
        else:
            encoded_forward = encoded[0][-1]
            encoded_backward = encoded[1][-1]

        forward_out = self.decoder(encoded_forward)
        backward_out = self.decoder(encoded_backward)

        return (forward_out, backward_out), state, encoded_raw, encoded_dropped
