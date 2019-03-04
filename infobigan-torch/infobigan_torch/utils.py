# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utility functions
"""


eps = 1e-15


def _listify(item, length=False):
    if length:
        if not (isinstance(item, tuple) or isinstance(item, list)):
            return [item] * length
        else:
            return list(item)
    else:
        if not (isinstance(item, tuple) or isinstance(item, list)):
            return [item]
        else:
            return list(item)
