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

## Model structure
![Model](Figure/Model.png)

##  Apply the model
To classify the music file, run

`python3 runner_cnn.py cli -i='path/to/target_file.wav'`

To classify music files in batch, put all files in batch list folder and run

`python3 runner_cnn.py batch`

## Accuracy
At Epoch 400, training on Kaggle GPU:
|  | **Loss**  | **Accuracy** | 
| ----- | ---- | ----- |
| Training   | `0.2018`  | `0.9600` |
| Test | `0.9225`  | `0.7033` |

Confusion matrix and model accuuracy:
| **null**                 | **blues** | **classical** | **country** | **disco** | **hiphop** | **jazz** | **metal** | **pop** | **reggae** | **rock** | **Precision/User’s ** |
|:------------------------:|:---------:|:-------------:|:-----------:|:---------:|:----------:|:--------:|:---------:|:-------:|:----------:|:--------:|:---------------------:|
| **blues**                | 18        | 0             | 4           | 2         | 0          | 0        | 1         | 0       | 1          | 1        | 66.67%                |
| **classical**            | 0         | 19            | 2           | 0         | 0          | 0        | 0         | 0       | 0          | 0        | 90.48%                |
| **country**              | 2         | 0             | 20          | 0         | 0          | 1        | 1         | 2       | 1          | 6        | 60.61%                |
| **disco**                | 0         | 0             | 1           | 18        | 1          | 0        | 1         | 0       | 1          | 7        | 62.07%                |
| **hiphop**               | 0         | 0             | 0           | 0         | 28         | 0        | 1         | 0       | 5          | 0        | 82.35%                |
| **jazz**                 | 1         | 4             | 0           | 0         | 0          | 21       | 0         | 0       | 0          | 2        | 75.00%                |
| **metal**                | 0         | 0             | 0           | 2         | 0          | 0        | 27        | 0       | 0          | 2        | 87.10%                |
| **pop**                  | 0         | 0             | 6           | 1         | 1          | 1        | 0         | 23      | 0          | 2        | 67.65%                |
| **reggae**               | 1         | 0             | 0           | 1         | 4          | 1        | 0         | 0       | 15         | 4        | 57.69%                |
| **rock**                 | 2         | 0             | 8           | 2         | 0          | 0        | 2         | 0       | 1          | 22       | 59.46%                |
| **Precision/Producer’s** | 75.00%    | 82.61%        | 48.78%      | 69.23%    | 82.35%     | 87.50%   | 81.82%    | 92.00%  | 62.50%     | 47.83%   | 70.33%                |




