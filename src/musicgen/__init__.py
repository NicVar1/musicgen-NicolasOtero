import torch
from scipy.io.wavfile import write
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODEL_ID = "facebook/musicgen-small"
DEVICE = "cpu"

PROMPT = "Indie video game soundtrack in the style of Undertale and Deltarune. Chiptune and 16-bit retro sound with square wave and triangle wave synths, catchy and memorable lead melody, bouncy bass line, simple punchy drum machine beat, and soft piano and string accents. Nostalgic, quirky and emotional mood, mixing a cheerful adventurous feeling with a touch of melancholy, like exploring a mysterious underground world full of strange and friendly characters. Upbeat tempo around 120 BPM, looping game music structure, lo-fi SNES-era sound, retro RPG battle and overworld theme. No vocals, no lyrics, instrumental only."

print("Cargando Musicgen...")

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)

model = model.to(DEVICE)
model.eval()

print("Preparando prompt...")

inputs = processor(text=[PROMPT], padding=True, return_tensors = "pt").to(DEVICE)

print("Generando musica...")

with torch.inference_mode():
    audio = model.generate(
        **inputs,
        do_sample=True,
        guidance_scale=3.0,
        max_new_tokens=2500
    )
sample_rate = model.config.audio_encoder.sampling_rate
waveform = audio[0,0].cpu().float().numpy()

write("musica.wav", sample_rate, waveform)
print("Musica generada")