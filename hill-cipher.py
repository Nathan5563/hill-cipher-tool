import math
import numpy as np

alphabet_upper = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
alphabet_lower = 'abcdefghijklmnopqrstuvwxyz'


# noinspection PyUnboundLocalVariable
def letter_to_num(letter):
    if letter in alphabet_upper:
        num = alphabet_upper.index(letter)
    elif letter in alphabet_lower:
        num = alphabet_lower.index(letter)

    return num


def num_to_letter(num):
    letter = alphabet_upper[num]
    return letter


def key_to_matrix(key):
    key_array = np.array([])

    # Check if key is a square matrix
    if int(math.sqrt(len(key))) == math.sqrt(len(key)):
        # Find the dimensions of the square matrix
        n = int(math.sqrt(len(key)))
        for i in key:
            key_array = np.append(key_array, [letter_to_num(i)])
        key_matrix = key_array.reshape(n, n)
        return key_matrix, n
    else:
        return 0


def string_to_matrix(string, key):
    string_array = np.array([])
    split_at = []
    loop_range = 1
    n = key_to_matrix(key)[1]

    for i in string:
        string_array = np.append(string_array, [letter_to_num(i)])

    # Find the indices at which the array should be split to form 1xn arrays
    for i in range(1, len(string_array)):
        if i % n == 0:
            split_at.append(i)
            loop_range += 1

    split_array = np.split(string_array, split_at)

    # Check if the last 1xn array is complete, and if not, add the letter 'Z' (index 25) to fill the remaining spaces
    leftover_difference = n - len(split_array[-1])
    for i in range(leftover_difference):
        split_array[-1] = np.append(split_array[-1], [25])

    return split_array, loop_range


def encrypt(plaintext, key):
    ciphertext = ''
    formatted_key, n = key_to_matrix(key)
    formatted_plaintext, loop_range = string_to_matrix(plaintext, key)

    # Only run if key is a square matrix
    if not isinstance(formatted_key, int):
        # Perform matrix multiplication and modulo 26 a number of times equal to the number of matrices
        for i in range(loop_range):
            # 2x1 arrays for n=2, 3x1 arrays for n=3, etc.
            plaintext_matrix = formatted_plaintext[i].reshape(n, 1)
            key_text_product = np.matmul(formatted_key, plaintext_matrix)
            ciphertext_matrix = key_text_product % 26

            # Convert the resulting nx1 matrix back into a 1xn array
            ciphertext_array = np.squeeze(ciphertext_matrix)

            # Convert the 1xn array into cipher text
            for j in ciphertext_array:
                ciphertext += f'{num_to_letter(int(j))}'

        return ciphertext
    else:
        print("Invalid key: Must be a square matrix")


def decrypt(ciphertext, key):
    plaintext = ''
    formatted_key, n = key_to_matrix(key)
    formatted_ciphertext, loop_range = string_to_matrix(ciphertext, key)

    if not isinstance(formatted_key, int):
        # Fnd the determinant of the key matrix and its inverse modulo 26
        mod_det = int(np.linalg.det(formatted_key) % 26)
        inv_mod_det = pow(mod_det, -1, 26)

        # Find the adjoint of the key matrix and take modulo 26
        adj = (np.linalg.inv(formatted_key) * np.linalg.det(formatted_key)) % 26

        # Find the inverse key matrix
        inverse_key_matrix = (inv_mod_det * adj) % 26

        # Perform matrix multiplication and modulo 26 a number of times equal to the number of matrices
        for i in range(loop_range):
            # 2x1 arrays for n=2, 3x1 arrays for n=3, etc.
            ciphertext_matrix = formatted_ciphertext[i].reshape(n, 1)
            key_text_product = np.matmul(inverse_key_matrix, ciphertext_matrix)

            # Add 0.5 to the index because it kept returning 26 % 26 = 0, a hacky solution, but it works
            plaintext_matrix = ((key_text_product % 26) + 0.5) % 26

            # Convert the resulting nx1 matrix back into a 1xn array
            plaintext_array = np.squeeze(plaintext_matrix)

            # Convert the 1xn array into plain text
            for j in plaintext_array:
                plaintext += f'{num_to_letter(int(j))}'

        return plaintext
    else:
        print("Invalid key: Must be a square matrix")


print(encrypt("HelloWorld", "DFCD"))
print(decrypt("PAKDWQXBWF", "DFCD"))