"""
Configuration file.
"""

USE_CUDA = True
DEVICES = [3]
CUDA_DEVICE = DEVICES[0]
VERSION = 1
MAXLEN = 30
BEST_MODEL = "../dir_HugeFiles/prev_model/skip-best-loss10.237"
WORD_DICT = '../dir_HugeFiles/instructions/skip_instruction.csv.pkl'