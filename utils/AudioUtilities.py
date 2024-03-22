# General Imports

import pandas as pd 
import numpy as np 
import sys
import os
import os.path
import soundfile as sf
import librosa

class AudioUtilities:
    def __init__(self):
        pass

    def export_song(self, data, sr, path):
        # Writes a song from the librosa format
        sf.write(path, data, sr, subtype='PCM_24')

    def speed_up_song(self, data, sr, speed=2):
        # Speeds up song based off of the data and sample rate
        return data, sr*speed

    def reverse_song(self, data, sr):
        # Reverses a song in librosa format
        return data[::-1], sr
    
    def process_music(self, file_path, song_offset):
        # Error Handling
        if not os.path.isfile(file_path):
            raise Exception("Please input a valid filepath")
        
        # General File Saving
        song_name = file_path.split('/')[-1]
        temp_file_path = file_path.replace(song_name, '')
        song_name = song_name.replace('.wav', '')
        song_name = song_name.replace('.mp3', '')

        data,sr=librosa.load(file_path, duration=3, offset=song_offset)
        s_data,s_sr=librosa.load(file_path, duration=6, offset=song_offset)

        # Add Effects
        speed_data, speed_sr = self.speed_up_song(s_data, s_sr)
        reverse_data, reverse_sr = self.reverse_song(data, sr)

        # Write Files
        new_path = temp_file_path + 'processed_' + song_name + '.wav'
        speed_file_path = temp_file_path + 'speed_up_' + song_name + '.wav'
        reversed_file_path = temp_file_path + 'reversed_' + song_name + '.wav'

        sf.write(new_path, data, sr, subtype='PCM_24')
        sf.write(speed_file_path, speed_data, speed_sr, subtype='PCM_24')
        sf.write(reversed_file_path, reverse_data, reverse_sr, subtype='PCM_24')

        print("Successfully manipulated audio.")