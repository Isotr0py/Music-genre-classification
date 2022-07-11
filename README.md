# Music-genre-classification
A conv1d neural network for music genre classification.
- Classify music files (mp3, wav and flac) based on 10 genres from GTZAN dataset.
- Use multiple layers of Conv1d Neural Nets and train model on Kaggle.
- Developed with Keras.

## Audio features extracted
Totally 40 features
- Chroma (12 features)
- Spectral Centroid (1 feature)
- MFCC (20 features)
- Spectral Contrast (7 features)

## Dependencies
All denpendencies for trainning model has fully provided in Kaggle environment. 

To run the model in local, listed denpendencies should be installed:
- numpy
- scipy -> for resample
- scikit-learn -> for data split and normalize
- librosa -> for feature extraction
- keras
    - `pip install keras`

To install all dependencies, run `pip install -r requirements.txt`

##  Apply the model
To classify the music file, run

`python3 runner_cnn.py cli 'path/to/target_file.wav'`

To classify music files in batch, put all files in batch list folder and run

`python3 runner_cnn.py batch`

## Accuracy
At Epoch 400, training on Kaggle GPU:
|  | **Loss**  | **Accuracy** | 
| ----- | ---- | ----- |
| Training   | `0.1724`  | `0.9357` |
| Test | `0.8963`  | `0.7200` |


