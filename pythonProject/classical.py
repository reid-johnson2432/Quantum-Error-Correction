import galois
import numpy as np


class ReedSolomon:
    def __init__(self, n, k):
        """
        Parameters:
        k: dimension of RS code
        q: Order of the Finite (galois) Field used to define RS alphabet
        n: length of RS codeword
        d: minimum distance between RS codewords
        t: error correcting ability (ECA) of classical RS code
        """
        self.galoisField = galois.GF(2 ** k)
        self.prime_subfield = self.galoisField.prime_subfield
        self.primitive_element = self.galoisField.primitive_element
        self.code = galois.ReedSolomon(n, k)
        self.length = self.code.n
        self.dimension = self.code.k
        self.minimum_distance = self.code.d
        self.ECA = self.code.t
        self.DFT = self.create_DFT()

    def generate_message(self):
        return self.galoisField.Random(self.dimension)

    def encode(self, message):
        return self.code.encode(message)

    def create_DFT(self):
        ones_arr = np.ones(self.length, dtype='int')
        alpha_mat = self.primitive_element * self.galoisField.Ones((self.length, self.length))
        for i in range(self.length):
            alpha_mat[i] = alpha_mat[i] ** (i * ones_arr)
            alpha_mat[:, i] = alpha_mat[:, i] ** (i * ones_arr)
        DFT = alpha_mat

        return DFT


def convert_to_GF2(string):
    GF2_array = galois.GF2([int(bit) for bit in string])
    return GF2_array
