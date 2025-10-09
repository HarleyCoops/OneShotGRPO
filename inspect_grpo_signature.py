import os
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')
os.environ.setdefault('TRANSFORMERS_NO_TENSORFLOW', '1')
os.environ.setdefault('USE_TF', '0')
from inspect import signature
from trl import GRPOTrainer
print(signature(GRPOTrainer.__init__))

