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
from mxnet import gluon
import gluonnlp as nlp

def testActivationRegularizationLoss():
    ar = nlp.loss.ActivationRegularizationLoss(2)
    print(ar)
    ar(*[mx.nd.arange(1000).reshape(10, 10, 10),
         mx.nd.arange(1000).reshape(10, 10, 10)])

def testTemporalActivationRegularizationLoss():
    tar = nlp.loss.TemporalActivationRegularizationLoss(1)
    print(tar)
    tar(*[mx.nd.arange(1000).reshape(10, 10, 10),
          mx.nd.arange(1000).reshape(10, 10, 10)])

def testJointActivationRegularizationLoss():
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    ar = nlp.loss.ActivationRegularizationLoss(2)
    tar = nlp.loss.TemporalActivationRegularizationLoss(1)
    jar = nlp.loss.JointActivationRegularizationLoss(loss, ar, tar)
    print(jar)
    jar(mx.nd.arange(1000).reshape(100, 10, 1),
        mx.nd.arange(1000).reshape(10, 10, 10),
        mx.nd.arange(1000).reshape(10, 10, 10),
        mx.nd.arange(1000).reshape(10, 10, 10))