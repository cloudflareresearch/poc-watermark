# Copyright (c) 2025 Cloudflare, Inc. All rights reserved.

from prc import *

print('generating key...')
key = key_gen()

codeword = encode(key)
assert detect(key, codeword)
print('detect: success')

codeword = encode(key)
for i in range(10):
    codeword[i] ^= True
assert detect(key, codeword)
print('detect with a few flipped bits: success')

codeword = encode(key)
for i in range(len(codeword) // 3):
    codeword[i] ^= True
assert detect(key, codeword)
print('detect with 1/3 bits flipped: success')

codeword = encode(key)
for i in range(len(codeword) // 2):
    codeword[i] ^= True
assert not detect(key, codeword)
print('detect with 1/2 bits flipped: failed as expected')
