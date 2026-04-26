#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import torch

import utils.register as R
from data.bioparse.hierarchy import Atom, Block

from .base import BaseTemplate


@R.register('LinearPeptide')
class LinearPeptide(BaseTemplate):

    def __init__(self, size_min=None, size_max=None, term_rel_pos=None):
        super().__init__(size_min, size_max)
        if term_rel_pos is None:
            self.term_rel_pos = None
        else:
            self.term_rel_pos = np.asarray(term_rel_pos, dtype=np.float32)
            if self.term_rel_pos.shape != (15,):
                raise ValueError('term_rel_pos must be a 15-dimensional CFG condition')

    def dummy_lig_block_bonds(self, cplx_desc, size):
        blocks = [Block(
            name='GLY', # use glycine instead of UNK so that is_aa can be identified as True
            atoms=[Atom(name='C', coordinate=[0, 0, 0], element='C', id=-1)],
            id=(i + 1, '')
        ) for i in range(size)]
        bonds = []
        return blocks, bonds

    def set_control(self, data):
        if self.term_rel_pos is not None:
            term_rel_pos = torch.as_tensor(
                self.term_rel_pos,
                dtype=data['term_rel_pos'].dtype,
                device=data['term_rel_pos'].device
            )
            data['term_rel_pos'].zero_()
            data['term_rel_pos'][data['generate_mask']] = term_rel_pos
        return data
