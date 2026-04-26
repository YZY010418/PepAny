#!/usr/bin/python
# -*- coding:utf-8 -*-
import pandas as pd
import torch


def read_csv(fpath, sep=','):
    heads, entries = [], []
    df = pd.read_csv(fpath, sep=sep)
    heads = list(df.columns)
    for rid in range(len(df)):
        entry = []
        for h in heads:
            entry.append(str(df[h][rid]))
        entries.append(entry)
    return heads, entries


def torch_load_full(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
