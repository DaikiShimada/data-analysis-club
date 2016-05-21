#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sklearn

import preprocess

data = preprocess.SanFranciscoCrimeData()
y = data.get_category_matrix()
print y.shape
