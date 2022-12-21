import struct
import numpy as np

def dump_numbers(filename, numbers, n, d):
  fout = open(filename, 'wb')
  fout.write(struct.pack('<i', n))
  fout.write(struct.pack('<i', d))
  for f in numbers:
      fout.write(struct.pack('<d', f))
  fout.close()


x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).astype(float)
dump_numbers("./Image_data/test.dat", x, 5,3)