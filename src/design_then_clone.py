import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import gc

output_file = "/workspace/data/audio/outputs/design_then_clone_output.wav"
# create a reference audio in the target style using the VoiceDesign model

design_model = Qwen3TTSModel.from_pretrained(
    "/workspace/qwen/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

ref_text = "H-hey! You dropped your... uh... calculus notebook? I mean, I think it's yours? Maybe?"
ref_instruct = "Male, 17 years old, tenor range, gaining confidence - deeper breath support now, though vowels still tighten when nervous"
ref_wavs, sr = design_model.generate_voice_design(
    text=ref_text,
    language="English",
    instruct=ref_instruct
)
sf.write("voice_design_reference.wav", ref_wavs[0], sr)


###################
# 1. Manually break the reference
if 'design_model' in locals():
    # If .cpu() is missing, we skip it and go straight to deletion
    del design_model

# 2. Crucial: Clear any 'ghost' references in the Python Garbage Collector
for _ in range(3): # Running it a few times ensures nested tensors are caught
    gc.collect()

# 3. Force the GPU to release the memory back to the OS
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# 4. Verification (Optional but helpful)
print(f"GPU Memory after clearing: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
###################


# build a reusable clone prompt from the voice design reference
clone_model = Qwen3TTSModel.from_pretrained(
    "/workspace/qwen/models/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

voice_clone_prompt = clone_model.create_voice_clone_prompt(
    ref_audio=(ref_wavs[0], sr),   # or "voice_design_reference.wav"
    ref_text=ref_text,
)

sentences = [
    "No problem! I actually... kinda finished those already? If you want to compare answers or something...",
    "What? No! I mean yes but not like... I just think you're... your titration technique is really precise!",
]

# sentences = [
#     "Four score and seven years ago our fathers brought forth on this continent a new nation, conceived in liberty, and dedicated to the proposition that all men are created equal.",
#     "Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battlefield of that war.",
#     "We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live.",
#     "It is altogether fitting and proper that we should do this.",
#     "But, in a larger sense, we can not dedicate -- we can not consecrate -- we can not hallow -- this ground.",
#     "The brave  men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract.",
#     "The world will little note, nor long remember what we say here, but it can never forget what they did here.",
#     "It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced.",
#     "It is rather for us to be here dedicated to the great task remaining before us -- that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion -- that we here highly resolve that these dead shall not have died in vain -- that this nation, under God, shall have a new birth of freedom -- and that government of the people, by the people, for the people, shall not perish from the earth.",
# ]

# reuse it for multiple single calls
wavs, sr = clone_model.generate_voice_clone(
    text=sentences[0],  
    language="English",
    voice_clone_prompt=voice_clone_prompt,
)
sf.write("clone_single_1.wav", wavs[0], sr)

wavs, sr = clone_model.generate_voice_clone(
    text=sentences[1],
    language="English",
    voice_clone_prompt=voice_clone_prompt,
)
sf.write("clone_single_2.wav", wavs[0], sr)

# or batch generate in one call
wavs, sr = clone_model.generate_voice_clone(
    text=sentences,
    language=["English", "English"],
    voice_clone_prompt=voice_clone_prompt,
)
for i, w in enumerate(wavs):
    sf.write(f"clone_batch_{i}.wav", w, sr)