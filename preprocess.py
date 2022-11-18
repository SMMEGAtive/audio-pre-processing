import os
import pickle
import librosa
import numpy as np

class AudioLoad:
    def __init__(self, sample_rate, duration, is_mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.is_mono = is_mono
        
    def load(self, filepath):
        signal = librosa.load(filepath, sr = self.sample_rate, duration = self.duration, mono = self.is_mono)[0]
        return signal

class ArrayPad:
    def __init__(self, mode="constant"):
        self.mode = mode
        
    def padLeft(self, array, n_missing):
        padded_array = np.pad(array, (n_missing, 0), mode = self.mode)
        return padded_array
    
    def padRight(self, array, n_missing):
        padded_array = np.pad(array, (0, n_missing), mode = self.mode)
        return padded_array

class SpectogramExtractor:
    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length
    
    def extract(self, signal):
        stft = librosa.stft(signal, n_fft = self.frame_size, hop_length = self.hop_length)[:-1]
        spectogram = np.abs(stft)
        log_spectogram = librosa.amplitude_to_db(spectogram)
        return log_spectogram
    
class MinMaxNormalizer:
    def __init__(self, min_value, max_value):
        self.min = min_value
        self.max = max_value
    
    def normalize(self, array):
        normalized_array = (array - array.min()) / (array.max() - array.min())
        normalized_array = normalized_array * (self.max - self.min) + self.min
        return normalized_array
    
    def denormalize(self, normalized_array, origin_min, origin_max):
        array = (normalized_array - self.min) / (self.max - self.min)
        array = array * (origin_max - origin_min) + origin_min
        return array

class SpectroSave:
    def __init__(self, save_dir, min_max_values_dir):
        self.save_dir = save_dir
        self.min_max_values_dir = min_max_values_dir
    
    def save(self, feature, filepath):
        save_path = self._generateSavePath(filepath)
        np.save(save_path, feature)

    def saveMinMaxValues(self, min_max_values):
        save_path = os.path.join(self.min_max_values_dir, "min_max_values.pkl")
        self._save(min_max_values, save_path)
    
    def _generateSavePath(self, filepath):
        filename = os.path.split(filepath)[1]
        save_path = os.path.join(self.save_dir, filename + ".npy")
        return save_path
    
    def _save(self, data, save_path):
        with open(save_path, "wb") as file:
            pickle.dump(data, file)

class ProcessAudio:
    def __init__(self):
        self.pad = None
        self.extractor = None
        self.normalizer = None
        self.spectrosave = None
        self.min_max_values = {}
        self._loader = None
        self._n_expected_samples = None
    
    @property
    def loader(self):
        return self._loader
    
    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._n_expected_samples = int(loader.sample_rate * loader.duration)
            
    def process(self, dir):
        for root, _, files in os.walk(dir):
            for file in files:
                if not file.endswith('.txt'):
                    filepath = os.path.join(root, file)
                    self._processFile(filepath)
                    print(f"Processed file {filepath}")
        self.spectrosave.saveMinMaxValues(self.min_max_values)

    def _processFile(self, filepath):
        signal = self.loader.load(filepath)
        if self._isNeedPadding(signal):
            signal = self._applyPadding(signal)
        feature = self.extractor.extract(signal)
        feature_normalized = self.normalizer.normalize(feature)
        save_path = self.spectrosave.save(feature_normalized, filepath)
        self._storeMinMaxValues(save_path, feature.min(), feature.max())
    
    def _isNeedPadding(self, signal):
        if len(signal) < self._n_expected_samples:
            return True
        return False
    
    def _applyPadding(self, signal):
        n_missing = self._n_expected_samples - len(signal)
        signal = self.pad.padLeft(signal, self._n_expected_samples)
        return signal
    
    def _storeMinMaxValues(self, save_path, min_val, max_val):
        self.min_max_values[save_path] = { "min_value": min_val, "max_value": max_val }

if __name__ == "__main__":
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    DURATION = 0.74
    SAMPLE_RATE = 22050
    MONO = True
    
    # You should replace these with your own directories 
    FILES_DIR = "D:\Individual\LibriSpeech\dev-clean"
    SPECTOGRAMS_SAVE_DIR = "D:\Individual\LibriSpeech\spectogram"
    MIN_MAX_VALUES_DIR = "D:\Individual\LibriSpeech\minmax_val"

    loader = AudioLoad(SAMPLE_RATE, DURATION, MONO)
    pad = ArrayPad()
    extractor = SpectogramExtractor(FRAME_SIZE, HOP_LENGTH)
    normalizer = MinMaxNormalizer(0, 1)
    spectrosave = SpectroSave(SPECTOGRAMS_SAVE_DIR, MIN_MAX_VALUES_DIR)

    preprocess = ProcessAudio()
    preprocess.loader = loader
    preprocess.pad = pad
    preprocess.extractor = extractor
    preprocess.normalizer = normalizer
    preprocess.spectrosave = spectrosave

    preprocess.process(FILES_DIR)

    print("Training Success!!")