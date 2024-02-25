import numpy as np


def letter_to_num(letter):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    num = alphabet.index(letter)
    return num


def num_to_letter(num):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    letter = alphabet[num]
    return letter


# Currently only works with a 3x3 (9 letter) key
def key_to_matrix(key):
    key_array = np.array([])

    for i in key:
        key_array = np.append(key_array, [letter_to_num(i)])
    key_matrix = key_array.reshape(3, 3)

    return key_matrix


def string_to_matrix(string):
    string_array = np.array([])
    split_at = []
    loop_range = 1

    for i in string:
        string_array = np.append(string_array, [letter_to_num(i)])

    # Find the indices at which the array should be split to form 1x3 arrays
    for i in range(1, len(string_array)):
        if i % 3 == 0:
            split_at.append(i)
            loop_range += 1

    split_array = np.split(string_array, split_at)

    # Check if the last 1x3 array is complete, and if not, add the letter 'Z' (index 25) to fill the remaining spaces
    leftover_difference = 3 - len(split_array[-1])
    for i in range(leftover_difference):
        split_array[-1] = np.append(split_array[-1], [25])

    return split_array, loop_range


def encrypt(plaintext, key):
    # Defining variables
    ciphertext = ''
    formatted_key = key_to_matrix(key)
    formatted_plaintext, loop_range = string_to_matrix(plaintext)

    # Perform matrix multiplication and modulo 26 a number of times equal to the number of matrices
    for i in range(loop_range):
        plaintext_matrix = formatted_plaintext[i].reshape(3, 1)
        key_text_product = np.matmul(formatted_key, plaintext_matrix)
        ciphertext_matrix = key_text_product % 26

        # Convert the resulting 3x1 matrix back into a 1x3 array
        ciphertext_array = np.squeeze(ciphertext_matrix)

        # Convert the 1x3 array into cipher text
        for j in ciphertext_array:
            ciphertext += f'{num_to_letter(int(j))}'

    return ciphertext


def decrypt(ciphertext, key):
    # Defining variables
    plaintext = ''
    formatted_key = key_to_matrix(key)
    formatted_ciphertext, loop_range = string_to_matrix(ciphertext)

    # Fnd the determinant of the key matrix and its inverse modulo 26
    mod_det = int(np.linalg.det(formatted_key) % 26)
    inv_mod_det = pow(mod_det, -1, 26)

    # Find the adjoint of the key matrix and take modulo 26
    adj = (np.linalg.inv(formatted_key) * np.linalg.det(formatted_key)) % 26

    # Find the inverse key matrix
    inverse_key_matrix = (inv_mod_det * adj) % 26

    # Perform matrix multiplication and modulo 26 a number of times equal to the number of matrices
    for i in range(loop_range):
        ciphertext_matrix = formatted_ciphertext[i].reshape(3, 1)
        key_text_product = np.matmul(inverse_key_matrix, ciphertext_matrix)

        # Add 0.5 to the index because it kept returning 26 % 26 = 0, a hacky solution, but it works
        plaintext_matrix = ((key_text_product % 26) + 0.5) % 26

        # Convert the resulting 3x1 matrix back into a 1x3 array
        plaintext_array = np.squeeze(plaintext_matrix)

        # Convert the 1x3 array into plain text
        for j in plaintext_array:
            plaintext += f'{num_to_letter(int(j))}'

    return plaintext


print(encrypt("HELLOWORLD", "GYBNQKURP"))
print(decrypt("TFJIPIJSGTNC", "GYBNQKURP"))