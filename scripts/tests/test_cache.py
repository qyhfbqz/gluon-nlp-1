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

import mxnet as mx
from gluonnlp.model import get_model
from ..language_model.cache import CacheCell


def test_cache():
    language_models = ['awd_lstm_lm_1150', 'awd_lstm_lm_600', 'standard_lstm_lm_200',
                   'standard_lstm_lm_650', 'standard_lstm_lm_1500']
    datasets = ['wikitext-2']

    for name in language_models:
        for dataset_name in datasets:
            pretrained_lm, vocab = get_model(name, dataset_name, pretrained=True, root='tests/data/model/')
            cache_cell = CacheCell(pretrained_lm, len(vocab), 10, 0.5, 0.5)
            outs, word_history, cache_history, hidden = \
                cache_cell(mx.nd.arange(10).reshape(10, 1), mx.nd.arange(10).reshape(10, 1), None, None)
            print(cache_cell)
            print("outs:")
            print(outs)
            print("word_history:")
            print(word_history)
            print("cache_history:")
            print(cache_history)
