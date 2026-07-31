import torch
import sys

try:
    ckpt = torch.load("ml_models/ckpt_iiitd11j3rru7wj576h92w37mpmjgtrc9y2vdyt3wkv6f/step_0000005/model/__0_0.distcp", weights_only=True)
    print("KEYS:", ckpt.keys())
    print("SUCCESS")
except Exception as e:
    print("ERROR:", e)
