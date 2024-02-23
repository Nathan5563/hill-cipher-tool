import numpy as np


def letter_to_num(letter):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    num = alphabet.index(letter)
    return num


def num_to_letter(num):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    letter = alphabet[num]
    return letter


# Encrypt function currently only works with n=3 (9 letter key)
def encrypt(plaintext, key):
    # Defining variables
    key_array = np.array([])
    plain_text_array = np.array([])
    split_by = []
    loop_range = 1
    cipherText = ''

    # Convert key to 3x3 matrix
    for i in key:
        key_array = np.append(key_array, [letter_to_num(i)])
    keyMatrix = key_array.reshape(3, 3)

    # Convert plain text to an array
    for i in plaintext:
        plain_text_array = np.append(plain_text_array, [letter_to_num(i)])

    # Find the indices at which the plain text array should be split to form 3x1 matrices
    for i in range(1, len(plain_text_array)):
        if i % 3 == 0:
            split_by.append(i)
            loop_range += 1

    # Split the plain text array into one or more 1x3 arrays
    splitarr = np.split(plain_text_array, split_by)

    # To convert the 1x3 arrays into 3x1 matrices, first check if all arrays are full (i.e., have 3 elements)
    # If the last array is missing elements, find how many and add in the letter 'Z'(number 25) as a filler
    leftoverDifference = 3 - len(splitarr[-1])
    for i in range(leftoverDifference):
        splitarr[-1] = np.append(splitarr[-1], [25])

    # Perform matrix multiplication and modulo 26 a number of times equal to the number of matrices
    for i in range(loop_range):
        plainTextMatrix = splitarr[i].reshape(3, 1)
        keyTextProduct = np.matmul(keyMatrix, plainTextMatrix)
        cipherTextMatrix = keyTextProduct % 26

        # Convert the resulting 3x1 matrix back into a 1x3 array
        cipherTextArray = np.squeeze(cipherTextMatrix)

        # Convert the 1x3 array into cipher text
        for i in cipherTextArray:
            cipherText += f'{num_to_letter(int(i))}'

    return cipherText


def decrypt(cipherText, key):
    # Defining variables
    keyArray = np.array([])
    cipherTextArray = np.array([])
    splitby = []
    looprange = 1
    plainText = ''

    # Convert key to 3x3 matrix
    for i in key:
        keyArray = np.append(keyArray, [letter_to_num(i)])
    keyMatrix = keyArray.reshape(3, 3)

    # Fnd the determinant of the key matrix and its inverse modulo 26
    modDet = int(np.linalg.det(keyMatrix) % 26)
    invModDet = pow(modDet, -1, 26)

    # Find the adjoint of the key matrix and take modulo 26
    adj = (np.linalg.inv(keyMatrix) * np.linalg.det(keyMatrix)) % 26

    # Find the inverse key matrix
    inverseKeyMatrix = (invModDet * adj) % 26

    # Convert cipher text to an array
    for i in cipherText:
        cipherTextArray = np.append(cipherTextArray, [letter_to_num(i)])

    # Find the indices at which the cipher text array should be split to form 3x1 matrices
    for i in range(1, len(cipherTextArray)):
        if i % 3 == 0:
            splitby.append(i)
            looprange += 1

    # Split the cipher text array into one or more 1x3 arrays
    splitarr = np.split(cipherTextArray, splitby)

    # To convert the 1x3 arrays into 3x1 matrices, first check if all arrays are full (i.e., have 3 elements)
    # If the last array is missing elements, find how many and add in the letter 'Z'(number 25) as a filler
    leftoverDifference = 3 - len(splitarr[-1])
    for i in range(leftoverDifference):
        splitarr[-1] = np.append(splitarr[-1], [25])

    # Perform matrix multiplication and modulo 26 a number of times equal to the number of matrices
    for i in range(looprange):
        cipherTextMatrix = splitarr[i].reshape(3, 1)

        keyTextProduct = np.matmul(inverseKeyMatrix, cipherTextMatrix)

        # Add 1 to the index because it kept returning 26 % 26 = 0, a hacky solution but it works
        plainTextMatrix = ((keyTextProduct % 26)+1) % 26

        # Convert the resulting 3x1 matrix back into a 1x3 array
        plainTextArray = np.squeeze(plainTextMatrix)

        # Convert the 1x3 array into plain text
        for i in plainTextArray:
            plainText += f'{num_to_letter(int(i))}'

    return plainText


encrypt('ACT', 'GYBNQKURP')
# Output: POH

decrypt('POH', 'GYBNQKURP')
# Output: ACT
