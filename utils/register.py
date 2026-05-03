#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Dict
from copy import deepcopy
import inspect

_NAMESPACE = {}
_CONSTRUCT_METADATA_KEYS = {
    'name',
    'pipeline',
    'n_samples',
    'max_sampling',
    'amide_bond_check',
    'sanitize_generation',
    'santinize_generation',
    'sanitize',
    'santinize',
    'resume_generation',
}

def register(name):
    def decorator(cls):
        assert name not in _NAMESPACE, f'Class {name} already registered'
        _NAMESPACE[name] = cls
        return cls
    return decorator


def get(name):
    if name not in _NAMESPACE:
        raise ValueError(f'Class {name} not registered')
    return _NAMESPACE[name]


def recur_construct(val: any):
    if isinstance(val, dict):
        if 'class' in val: return construct(val) # leaf node
        for key in val:
            val[key] = recur_construct(val[key])
    elif isinstance(val, list):
        val = [recur_construct(v) for v in val]
    return val


def construct(config: Dict, **kwargs):
    config = deepcopy(config)
    cls_name = config.pop('class')
    cls = get(cls_name)
    try:
        signature = inspect.signature(cls.__init__)
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )
        if not accepts_kwargs:
            for key in _CONSTRUCT_METADATA_KEYS:
                if key not in signature.parameters:
                    config.pop(key, None)
    except (TypeError, ValueError):
        pass
    config.update(kwargs)
    return cls(**config)
