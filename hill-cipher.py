import numpy as np

def letter_to_num(letter):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    num = alphabet.index(letter)
    return num

def num_to_letter(num):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    letter = alphabet[num]
    return letter

# Encrypt function currently only works with a 9 letter key and a 3 letter plainText
def encrypt(plainText, key):
    keyArray = np.array([])
    plainTextArray = np.array([])
    cipherText = ''

    for i in key:
        keyArray = np.append(keyArray, [letter_to_num(i)])
                
    keyMatrix = keyArray.reshape(3, 3)
    
    for i in plainText:
        plainTextArray = np.append(plainTextArray, [letter_to_num(i)])
        
    plainTextMatrix = plainTextArray.reshape(3, 1)
    
    keyTextProduct = np.matmul(keyMatrix, plainTextMatrix)
    
    cipherTextMatrix = keyTextProduct % 26
    
    cipherTextArray = np.squeeze(cipherTextMatrix)
    
    for i in cipherTextArray:
        cipherText += f'{num_to_letter(int(i))}'
            
    return cipherText