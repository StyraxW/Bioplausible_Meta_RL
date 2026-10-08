"""Bioplausible meta-RL models: shared infrastructure and Models 1-3.

The valuernn fork imports its own modules as top-level names
(`from model import ValueRNN`, `from tasks.inference import ...`),
so its directory has to be on sys.path.
"""
import os
import sys

VALUERNN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'valuernn')
if VALUERNN_DIR not in sys.path:
    sys.path.insert(0, VALUERNN_DIR)
