#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last Updated: 2022-07-11
Git Updated: True 
@author: Isotr0py
"""

import numpy as np
import librosa
import keras
from scipy import signal
from sklearn.preprocessing import StandardScaler
import os
from argparse import ArgumentParser
import warnings

warnings.filterwarnings('ignore')

# supported formats, to support .mp3, ffmpeg should be added to Environment Variables, if not may result errors for mp3 file
formats = ['mp3','flac','wav'] 

class audio_cls:
    # 10 classes genres
    genres = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
    model = keras.models.load_model('Audino_CNN')   # load model
    model.summary()                                 # display model

    def __init__(self) -> None:
        pass
    
    # load file and normalize feature
    def load(self,file)->np.ndarray:
        y,sr = librosa.load(file)
        feature = self.featureCal(y,sr,times_len=256)
        feature = np.stack([feature])
        feature = np.transpose(feature,[0,2,1])
        scaler = StandardScaler()
        for i in range(feature.shape[0]):
            feature[i,:,:] = scaler.fit_transform(feature[i,:,:])
        return feature

    # extract and resample feature
    def featureCal(self,y,sr,times_len)->np.array:
        chroma_stft = librosa.feature.chroma_stft(y,sr)
        spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y,sr,n_mfcc=20)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features = chroma_stft.copy()
        for feature in [spectral_center,mfcc,spectral_contrast]:
            features = np.append(features,feature,axis=0)
        features = signal.resample(features,times_len,axis=1)
        return features[:,:times_len]

    # classify genre
    def predict_genre(self,feature):
        res = self.model.predict(feature)
        genre = self.genres[np.argmax(res)]
        return genre

# batch mode
def run_batch(args):
    audino = audio_cls()
    for file in os.listdir("./Batch_list/"):
        if file.split(".")[-1] in formats:
            feature = audino.load(f"./Batch_list/{file}")
            genre = audino.predict_genre(feature)
            print(f"File:{file}  Genre:{genre}")

# cli mode
def run_cli(args):
    file = args.i
    if not file.split(".")[-1] in formats:
        raise ValueError("Unsupported audio format.")
    audino = audio_cls()
    feature = audino.load(f"{file}")
    genre = audino.predict_genre(feature)
    print(f"File:{file.split('/')[-1]}  Genre:{genre}")

# command line parser
def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()
    parser.set_defaults(func=lambda _: parser.print_usage())

    cli_parser = subparsers.add_parser("cli")   # add cli mode
    cli_parser.add_argument('-i', type=str, help='input audio file')
    cli_parser.set_defaults(func=run_cli)

    batch_parser = subparsers.add_parser("batch")   # add batch mode
    batch_parser.set_defaults(func=run_batch)
    # setting_parser = subparsers.add_parser("setting")
    # run_parser.set_defaults(func=lambda _:setting())
    args = parser.parse_args()
    args.func(args)

if __name__=="__main__":
    main()