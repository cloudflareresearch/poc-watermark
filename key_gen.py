# Copyright (c) 2025 Cloudflare, Inc. All rights reserved.

from prc import key_gen, key_to_pem

print('generating key pair. This may take a few seconds.')
key = key_gen()

key_file_name = 'test_key.pem'
print('writing key to {}...'.format(key_file_name))
with open(key_file_name, 'w') as f:
    f.write(key_to_pem(key))
