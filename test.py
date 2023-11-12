import torchaudio
audio, sample_rate = torchaudio.load('fixtures/audio/keanu.mp3', format='mp3')
print(audio.shape)
