import torch
import struct

def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])[2:]

scale_table = torch.load("model_calib.pth")
calibrator_cache = open("model.calib", "w")
calibrator_cache.write("TRT-7134-EntropyCalibration2\n")

for name in scale_table:
    calibrator_cache.write("{:s}: {}\n".format(
        name, float_to_hex(scale_table[name])))
