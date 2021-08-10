import pretty_midi
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import os, shutil

current_directory = os.getcwd()

midis_folder = current_directory + '/generated_midi_files/'
pianorolls_folder = current_directory + '/generated_pianoroll_files/'

if not os.path.exists(pianorolls_folder):
    os.makedirs(pianorolls_folder)

midi_file_names = os.listdir(midis_folder)

def plot_piano_roll(pm, start_pitch, end_pitch, fs=4100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(pm.get_piano_roll(fs = fs, times = np.arange(0, 2, 1./fs))[start_pitch:end_pitch], #  times = np.arange(0, get_end_time(), 1./fs)
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch),
                             )

counter = 0
total_number_of_files = len(midi_file_names)

for midi_file in midi_file_names:
    counter += 1
    print('Processing: ' + str(counter) + ' of ' + str(total_number_of_files))
    if midi_file[-4:] == '.mid':
        try: 
            pm = pretty_midi.PrettyMIDI(midis_folder + midi_file)
            for instrument in pm.instruments:
                # Don't want to shift drum notes
                for note in instrument.notes:
                    note.end -= 0.003 # To separate consecutive same pitches in the pianoroll display, otherwise appears as one long note

            plt.figure(figsize=(12, 6)) # 12 4
            plt.rcParams.update({'font.size': 17})
            plt.rcParams["font.weight"] = "bold"
            plt.rcParams["axes.labelweight"] = "bold"
            plot_piano_roll(pm, 53, 85)
            plt.savefig(pianorolls_folder + midi_file[:-4] + ".png")
        except IndexError:
            pass
