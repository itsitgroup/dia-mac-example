from dia.model import Dia
import torch

device = torch.device("cpu")

# Pass the device parameter directly when loading
model = Dia.from_pretrained(
    "nari-labs/Dia-1.6B",
    compute_dtype="float32",
    device=device
)

text = "[S1] I'm sorry… (crying) [S2] Please calm down."

output = model.generate(text, use_torch_compile=False, verbose=True)

model.save_audio("simple.mp3", output)
