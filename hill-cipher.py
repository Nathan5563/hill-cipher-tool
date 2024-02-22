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
def encrypt(plainText, key):
    # Defining variables
    keyArray = np.array([])
    plainTextArray = np.array([])
    splitby = []
    looprange = 1
    cipherText = ''

    # Convert key to 3x3 matrix
    for i in key:
        keyArray = np.append(keyArray, [letter_to_num(i)])     
    keyMatrix = keyArray.reshape(3, 3)

    # Convert plain text to an array
    for i in plainText:
        plainTextArray = np.append(plainTextArray, [letter_to_num(i)])

    # Find the indices at which the plain text array should be split to form 3x1 matrices
    for i in range(1, len(plainTextArray)):
        if i % 3 == 0:
            splitby.append(i)
            looprange += 1
            
    # Split the plain text array into one or more 1x3 arrays
    splitarr = np.split(plainTextArray, splitby)
    
    # To convert the 1x3 arrays into 3x1 matrices, first check if all arrays are full (i.e., have 3 elements)
    # If the last array is missing elements, find how many and add in the letter 'Z'(number 25) as a filler
    leftoverDifference = 3-len(splitarr[-1])
    for i in range(leftoverDifference):
        splitarr[-1] = np.append(splitarr[-1], [25])
            
    # Perform matrix multiplication and modulo 26 a number of times equal to the number of matrices
    for i in range(looprange):
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
        
    modDet = np.linalg.det(keyMatrix) % 26
    
    adj = (np.linalg.inv(keyMatrix) * np.linalg.det(keyMatrix)) % 26
    
    inverseKeyMatrix = (modDet * adj) % 26
    
    print(modDet) 
    print(inverseKeyMatrix)
    
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
    leftoverDifference = 3-len(splitarr[-1])
    for i in range(leftoverDifference):
        splitarr[-1] = np.append(splitarr[-1], [25])
        
    for i in range(looprange):
        cipherTextMatrix = splitarr[i].reshape(3, 1)
                
        keyTextProduct = np.matmul(inverseKeyMatrix, cipherTextMatrix)
        
        plainTextMatrix = keyTextProduct % 26
        
        print(plainTextMatrix)
        
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