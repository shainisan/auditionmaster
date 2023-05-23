from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mir_eval


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'

ALLOWED_EXTENSIONS = {'mp3'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audiofile' in request.files:
        audiofile = request.files['audiofile']
        if audiofile.filename == '':
            return 'No file selected'
        if audiofile and allowed_file(audiofile.filename):
            filename = secure_filename(audiofile.filename)
            audiofile.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return process_audio(filename)
        else:
            return 'Invalid file format'
    else:
        return 'No file part'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_audio(filename):
    # Load the audio file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    y, sr = librosa.load(file_path)

    # Prepare time array for plots
    times_array = librosa.frames_to_time(np.arange(len(y)), sr=sr)

    # pYIN pitch tracking
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

    # times corresponding to f0 estimates
    times = librosa.times_like(f0)

    # Beat tracking
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

    # Beat frames to time
    beat_times = librosa.frames_to_time(beats, sr=sr)

    # Calculate the time difference between beats
    beat_diff = np.diff(beat_times)

    # Convert f0 values into musical notes
    notes = mir_eval.melody.hz2cents(f0, base_frequency=442.0) / 100

    # Remove unvoiced parts
    notes = notes[voiced_flag]

    # Filter times with voiced_flag
    times_voiced = times[voiced_flag]

    # Compute the short-time Fourier transform (STFT)
    D = librosa.stft(y)

    # Separate the harmonic and percussive components
    H, P = librosa.decompose.hpss(D)

    # Convert the magnitude spectrogram to dB-scaled spectrogram
    H_db = librosa.amplitude_to_db(np.abs(H), ref=np.max)

    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    mfcc_std = np.std(mfccs, axis=1)

    # Compute "timbre score" 
    timbre_score = 1 / np.mean(mfcc_std)

    # Create subplots using gridspec
    fig = plt.figure(figsize=(14, 16))
    gs = gridspec.GridSpec(10, 1)

    # Waveform plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(times_array, y)
    ax1.set_title('Waveform')

    # RMS energy over time plot
    ax2 = fig.add_subplot(gs[1, 0])
    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    ax2.plot(rms_times, rms)
    ax2.set_title('Loudness (RMS Energy) Over Time')

    # Inter-beat intervals over time plot
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(beat_times[:-1], beat_diff)
    ax3.set_title('Tempo (Inter-beat Intervals) Over Time')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Interval (s)')

    # Pitch track plot
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.plot(times_voiced, notes, label='f0')
    ax4.legend(loc='upper right')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Pitch')
    ax4.set_title('Pitch track')
    ax4.grid(True)

    # Harmonic spectrogram plot
    ax5 = fig.add_subplot(gs[4:7, 0])
    librosa.display.specshow(H_db, sr=sr, x_axis='time', y_axis='log', ax=ax5)
    ax5.set_title('Harmonic Spectrogram')

    # MFCCs plot
    ax6 = fig.add_subplot(gs[7:10, 0])
    librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax6)
    ax6.set_title('MFCCs')

    # Save the combined image
    plots_folder = 'static/plots'
    os.makedirs(plots_folder, exist_ok=True)
    combined_path = f'{plots_folder}/combined_plots.png'
    plt.tight_layout(pad=0)  # tight_layout with padding set to 0
    plt.savefig(combined_path, bbox_inches='tight', pad_inches=0)  # saving figure with additional arguments
    #plt.close(fig)

    rms = librosa.feature.rms(y=y)[0]
    loudness_score = np.std(rms)
    beat_diff_score = 1 / np.std(beat_diff)
    pitch_score = 1 / np.std(notes)
    contrast = librosa.feature.spectral_contrast(S=np.abs(H), sr=sr)
    contrast_score = np.mean(contrast)

    total_score = (loudness_score + beat_diff_score + pitch_score + contrast_score + timbre_score) / 5

    scores = {
        'loudness_score': loudness_score,
        'beat_diff_score': beat_diff_score,
        'pitch_score': pitch_score,
        'contrast_score': contrast_score,
        'timbre_score': timbre_score
    }

    plots = {
        'combined_plots': combined_path
    }

    return render_template('results.html', scores=scores, plots=plots, total_score=total_score, filename=filename)


if __name__ == '__main__':
    app.run(debug=True)

