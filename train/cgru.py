import ctypes

import numpy as np
from .paths import *

from .ctokenizer import CTokenizer

class Matrix(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("m", ctypes.c_int),
        ("n", ctypes.c_int),
        ("stride", ctypes.c_int),
    ]


class CGRU:
    def __init__(
        self,
        weights_path,
        lib_path=BUILD / "libgru.so",
        tokenize=True,
    ):
        weights_path = Path(weights_path)
        lib_path = Path(lib_path)

        if not weights_path.exists():
            raise FileNotFoundError(f"{weights_path} not found")
        
        if not lib_path.exists():
            raise FileNotFoundError(f"{lib_path} not found")
        
        self.weights_path = weights_path
        self.libpath = lib_path

        self.lib = ctypes.CDLL(lib_path.as_posix())
        self.lib.gruCreate.argtypes = [ctypes.c_char_p]
        self.lib.gruCreate.restype = ctypes.POINTER(ctypes.c_void_p)

        self.lib.gruPredict.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # GRU obj
            ctypes.POINTER(ctypes.c_int),  # tokens
            ctypes.c_int,  # num_tokens
        ]
        self.lib.gruPredict.restype = ctypes.c_int

        self.lib.gruGetLastState.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # GRU obj
            ctypes.POINTER(ctypes.c_int),  # tokens
            ctypes.c_int,  # num_tokens
        ]
        self.lib.gruGetLastState.restype = Matrix

        self.lib.gruGetLogits.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # GRU obj
            ctypes.POINTER(ctypes.c_int),  # tokens
            ctypes.c_int,  # num_tokens
        ]
        self.lib.gruGetLogits.restype = Matrix

        self.gru = self.lib.gruCreate(weights_path.as_posix().encode())

        self.tokenize = tokenize
        if tokenize:
            self.tokenizer = CTokenizer()

    def get_last_state(self, tokens):
        if self.tokenize:
            tokens = self.tokenizer(tokens)

        tokens_array = (ctypes.c_int * len(tokens))()
        for i, token in enumerate(tokens):
            tokens_array[i] = token

        last_state = self.lib.getLastStateGRU(
            self.gru, tokens_array, ctypes.c_int(len(tokens))
        )
        last_state = np.ctypeslib.as_array(last_state.data, (last_state.n,))

        return last_state
    
    def get_logits(self, tokens):
        if self.tokenize:
            tokens = self.tokenizer(tokens)

        tokens_array = (ctypes.c_int * len(tokens))()
        for i, token in enumerate(tokens):
            tokens_array[i] = token

        logits = self.lib.gruGetLogits(
            self.gru, tokens_array, ctypes.c_int(len(tokens))
        )
        logits = np.ctypeslib.as_array(logits.data, (logits.n,))

        return logits

    def __call__(self, tokens):
        if self.tokenize:
            tokens = self.tokenizer(tokens)

        tokens_array = (ctypes.c_int * len(tokens))()
        for i, token in enumerate(tokens):
            tokens_array[i] = token

        prediction = self.lib.gruPredict(
            self.gru, tokens_array, ctypes.c_int(len(tokens))
        )

        return prediction
