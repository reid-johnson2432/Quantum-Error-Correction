from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.circuit.library import QFT
from random import randrange
from enum import Enum, auto
import numpy as np


# Define Errors
class ErrorTypes(Enum):
    x = auto()
    y = auto()
    z = auto()


class QuantumReedSolomon:
    def __init__(self, classical_code):
        self.classical_code = classical_code
        self.length = self.get_length()
        self.dimension = self.get_dimension()
        self.message_register = self.create_message_register()
        self.ancillia_register = self.create_ancillia_register()
        self.classical_register = self.create_classical_register()

    def get_dimension(self):
        N = self.classical_code.length
        K = self.classical_code.length - self.classical_code.minimum_distance + 1
        # dimension of quantum code Q
        if K > N / 2:
            print('dimQ: k*(2*K - N)')
            dimension = self.classical_code.dimension * (2 * K - N)
        else:
            dimension = self.classical_code.dimension * (N - 2 * K)

        return dimension

    def get_length(self):
        return self.classical_code.dimension * self.classical_code.length

    def create_message_register(self):
        mqubits = QuantumRegister(self.length)
        return mqubits

    def create_ancillia_register(self):
        ancqubits = QuantumRegister(2 * self.classical_code.dimension * self.dimension)
        return ancqubits

    def create_classical_register(self):
        cr = ClassicalRegister(2 * self.classical_code.dimension * self.dimension)
        return cr

    def convert_to_quantum(self, codeword):
        codeword_spectrum = codeword @ self.classical_code.DFT
        quantum_codeword = np.hstack([self.classical_code.galoisField.vector(element) for element in codeword_spectrum])

        return quantum_codeword


class QRSCircuit(QuantumCircuit):
    def __init__(self, QRS, mqubits, ancqubits, cr):
        super(QRSCircuit, self).__init__(mqubits, ancqubits, cr)
        self.Quantum_ReedSolomon = QRS
        self.message_register = mqubits
        self.ancillia_register = ancqubits
        self.classical_register = cr

    def initialize(self, quantum_codeword):
        [self.x(index) for index, value in enumerate(quantum_codeword[::-1]) if int(value) == 1]
        self.barrier()

    def encode(self):
        """
        This method will implement the encoding scheme as follows:
        --> Apply Hadamard gate to the last k*K message qubits
        --> Apply QFT to all message qubits
        """
        # Encoder
        classical_dim = self.Quantum_ReedSolomon.classical_code.dimension
        QRS_dim = self.Quantum_ReedSolomon.dimension
        QRS_length = self.Quantum_ReedSolomon.length
        [self.h(qubit) for qubit in self.message_register[classical_dim + classical_dim * QRS_dim:]]

        self.append(QFT(QRS_length).inverse(), self.message_register)  # apply qft to all codeword qubits
        self.barrier()

    def decode(self):
        """
        This method will decode ...
        """
        # Decoder
        self.append(QFT(self.Quantum_ReedSolomon.length), self.message_register)  # apply qft to all codeword qubits

        # Swap first kK ancillia bits with [k, k + kK] qubits
        classical_dim = self.Quantum_ReedSolomon.classical_code.dimension
        quantum_dim = self.Quantum_ReedSolomon.dimension
        for index, qubit in enumerate(self.ancillia_register[:classical_dim * quantum_dim]):
            self.cx(self.qubits[index], qubit)

        for qubit in self.message_register[classical_dim + classical_dim * quantum_dim:]:
            self.h(qubit)

        # Swap ancillia first kK ancillia bits with [k, k + kK] qubits
        step = classical_dim + classical_dim*quantum_dim
        for qubit in self.ancillia_register[classical_dim:2 * classical_dim * quantum_dim]:
            self.cx(self.qubits[step], qubit)
            step += 1

        for qubit in self.message_register[classical_dim + classical_dim * quantum_dim:]:
            self.h(qubit)

        self.append(QFT(self.Quantum_ReedSolomon.length).inverse(), self.message_register)

        self.barrier()

    def perform_measurement(self):
        """

        :return:
        """
        self.measure(self.ancillia_register, self.classical_register)
        self.draw(fold=-1)

    def scramble_qubits(self, num_errors):
        """
        Current Build only supports bit-flip errors
        :return:
        """
        for _ in range(num_errors):
            error_qubit = self.message_register[randrange(0, self.Quantum_ReedSolomon.length)]
            error_type = list(ErrorTypes)[0]
            if error_type == ErrorTypes.x:
                self.x(error_qubit)
            if error_type == ErrorTypes.y:
                self.y(error_qubit)
            if error_type == ErrorTypes.z:
                self.z(error_qubit)

