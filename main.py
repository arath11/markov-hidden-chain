import os
import matplotlib.pyplot as plt
import numpy as np
from hmmlearn import hmm
from python_speech_features import mfcc, logfbank
from scipy.io import wavfile

#mono -> 22050Hz a 16 bits es preferible usar WAV como formato de audio
#se usa MFCC para tener el espectro

#extract
def extract(filePath):
    sampling_freq, audio = wavfile.read(filePath)
    mfcc_features = mfcc(audio, sampling_freq)
    filterbank_features = logfbank(audio, sampling_freq)
    #ver como quedo
    print('\nMFCC:\nNumber of windows =', mfcc_features.shape[0])
    print('Length of each feature =', mfcc_features.shape[1])
    print('\nFilter bank:\nNumber of windows =', filterbank_features.shape[0])
    print('Length of each feature =', filterbank_features.shape[1])

#extract("./training/dog/PoodleLadrido.wav")
""""
def samples():
    genre_list = ["Electro","Pop","Reg"]
    print(len(genre_list))
    figure = plt.figure(figsize=(20, 3))
    for idx, genre in enumerate(genre_list):
        print(genre)
        #example_data_path = 'genres/'+genre
        #print(example_data_path)
        file_paths =genre+".wav"
        print(file_paths)
        sampling_freq, audio = wavfile.read(file_paths)
        mfcc_features = mfcc(audio, sampling_freq, nfft=1024)
        print(file_paths[0], mfcc_features.shape[0])
        plt.yscale('linear')
        plt.matshow((mfcc_features.T)[:, :300])
        plt.text(150, -10, genre, horizontalalignment='center', fontsize = 20)
        plt.yscale('linear')
        plt.show()
#samples()
"""

#Declare a HMM Trainer class
class HMMTrainer(object):
  def __init__(self, model_name='GaussianHMM', n_components=4, cov_type='diag', n_iter=1000):
    self.model_name = model_name
    self.n_components = n_components
    self.cov_type = cov_type
    self.n_iter = n_iter
    self.models = []
    if self.model_name == 'GaussianHMM':
      self.model = hmm.GaussianHMM(n_components=self.n_components,        covariance_type=self.cov_type,n_iter=self.n_iter)
    else:
      raise TypeError('Invalid model type') 

  def train(self, X):
    np.seterr(all='ignore')
    self.models.append(self.model.fit(X))
    # Run the model on input data
  def get_score(self, input_data):
    return self.model.score(input_data)

def training():
  #Training models for each type of animal sound
  #iterar sobre el folder que se escogio
  hmm_models = []
  input_folder = 'training/'
  # Parse the input directory
  for dirname in os.listdir(input_folder):
      # Get the name of the subfolder
      subfolder = os.path.join(input_folder, dirname)
      if not os.path.isdir(subfolder):
          continue
      # Extract the label
      label = subfolder[subfolder.rfind('/') + 1:]
      # Initialize variables
      X = np.array([])
      y_words = []
      # Iterate through the audio files (leaving 1 file for testing in each class)
      for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]:
              # Read the input file
              filepath = os.path.join(subfolder, filename)
              sampling_freq, audio = wavfile.read(filepath)
              # Extract MFCC features
              mfcc_features = mfcc(audio, sampling_freq)
              # Append to the variable X
              if len(X) == 0:
                  X = mfcc_features
              else:
                  X = np.append(X, mfcc_features, axis=0)

              # Append the label
              y_words.append(label)
      print('X.shape =', X.shape)
      # Train and save HMM model
      hmm_trainer = HMMTrainer(n_components=10)
      hmm_trainer.train(X)
      hmm_models.append((hmm_trainer, label))
      hmm_trainer = None

  return hmm_models


def classify(hmm_models):
  #Classify animal sound
  input_folder = 'test/'
  real_labels = []
  pred_labels = []
  for dirname in os.listdir(input_folder):
    subfolder = os.path.join(input_folder, dirname)
    if not os.path.isdir(subfolder):
      continue
    # Extract the label
    label_real = subfolder[subfolder.rfind('/') + 1:]
    
    #print([x for x in os.listdir(subfolder) if x.endswith('.wav')])
    for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')]:
      real_labels.append(label_real)
      #print(filename+"__________________")
      filepath = os.path.join(subfolder, filename)
      sampling_freq, audio = wavfile.read(filepath)
      mfcc_features = mfcc(audio, sampling_freq)
      max_score = -9999999999999999999
      output_label = None
      for item in hmm_models:
        hmm_model, label = item
        score = hmm_model.get_score(mfcc_features)
        print(score)
        if score > max_score:
            max_score = score
            output_label = label
            
      #  print(f'{score}___{label}')            
      pred_labels.append(output_label)
  return [pred_labels,real_labels]




def main():
  resultado=classify(training())
  os.system("cls")
  print(f'_______________________')
  print(f'|Predicción   |   Real|')
  print(f'_______________________')
  for i in range(len(resultado[0])):
    print(f'|     {resultado[0][i]}     |    {resultado[1][i]}|')
  print(f'_______________________')
main()