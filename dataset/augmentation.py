import torchaudio

class TimeMasking:
    def __init__(self, time_mask_param=30):
        self.op = torchaudio.transforms.TimeMasking(time_mask_param)

    def __call__(self, waveform):
        return self.op(waveform)

class FrequencyMasking:
    def __init__(self, freq_mask_param=15):
        self.op = torchaudio.transforms.FrequencyMasking(freq_mask_param)

    def __call__(self, waveform):
        return self.op(waveform)

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, waveform):
        for t in self.transforms:
            waveform = t(waveform)
        return waveform
