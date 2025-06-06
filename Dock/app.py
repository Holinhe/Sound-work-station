import os
from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from scipy.io import wavfile
import librosa
import io
import uuid
from datetime import datetime
import warnings  # Added import for warnings module

app = Flask(__name__)
UPLOAD_FOLDER = '/tmp/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'wav'}

# Store multiple audio files in memory
audio_files = {}  # {id: {'data': np.array, 'sample_rate': int, 'filename': str, 'timestamp': str}}
current_audio_id = None  # ID of the currently selected audio

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def normalize_audio(data):
    """Normalize audio to avoid clipping and excessive loudness."""
    max_amplitude = np.max(np.abs(data))
    if max_amplitude > 0:
        data = data / max_amplitude * 0.9  # Scale to 90% of max amplitude
    return data

@app.route('/')
def index():
    return send_file('static/index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global audio_files, current_audio_id
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        sample_rate, audio_data = wavfile.read(file_path)
        if len(audio_data.shape) == 1:
            audio_data = audio_data.astype(np.float32)[:, np.newaxis]
        else:
            audio_data = audio_data.astype(np.float32)
        audio_data = normalize_audio(audio_data)
        audio_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime('%H:%M:%S')
        audio_files[audio_id] = {'data': audio_data, 'sample_rate': sample_rate, 'filename': filename, 'timestamp': timestamp}
        if current_audio_id is None:
            current_audio_id = audio_id
        return jsonify({'message': 'File uploaded', 'id': audio_id, 'filename': f"{filename} ({timestamp})"}), 200
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/list', methods=['GET'])
def list_files():
    return jsonify([
        {'id': audio_id, 'filename': info['filename'] + f" ({info['timestamp']})"}
        for audio_id, info in audio_files.items()
    ])

@app.route('/select/<audio_id>', methods=['POST'])
def select_audio(audio_id):
    global current_audio_id
    if audio_id in audio_files:
        current_audio_id = audio_id
        return jsonify({'message': f'Selected audio {audio_files[audio_id]["filename"]} ({audio_files[audio_id]["timestamp"]})'}), 200
    return jsonify({'error': 'Audio not found'}), 404

@app.route('/play', methods=['GET'])
def play_audio():
    if current_audio_id is None or current_audio_id not in audio_files:
        return jsonify({'error': 'No audio selected'}), 400
    audio_data = audio_files[current_audio_id]['data']
    sample_rate = audio_files[current_audio_id]['sample_rate']
    output = io.BytesIO()
    wavfile.write(output, sample_rate, (audio_data * 32767).astype(np.int16))
    output.seek(0)
    return send_file(output, mimetype='audio/wav', as_attachment=True, download_name='current.wav')

@app.route('/download', methods=['GET'])
def download_audio():
    if current_audio_id is None or current_audio_id not in audio_files:
        return jsonify({'error': 'No audio selected'}), 400
    audio_data = audio_files[current_audio_id]['data']
    sample_rate = audio_files[current_audio_id]['sample_rate']
    filename = audio_files[current_audio_id]['filename'].rsplit('.', 1)[0] + '_edited.wav'
    output = io.BytesIO()
    wavfile.write(output, sample_rate, (audio_data * 32767).astype(np.int16))
    output.seek(0)
    return send_file(output, mimetype='audio/wav', as_attachment=True, download_name=filename)

@app.route('/trim', methods=['POST'])
def trim_audio():
    if current_audio_id is None or current_audio_id not in audio_files:
        return jsonify({'error': 'No audio selected'}), 400
    data = request.get_json()
    start_ms = int(data['start'])
    end_ms = int(data['end'])
    audio_data = audio_files[current_audio_id]['data']
    sample_rate = audio_files[current_audio_id]['sample_rate']
    start_sample = int(start_ms * sample_rate / 1000)
    end_sample = int(end_ms * sample_rate / 1000)
    if 0 <= start_sample < end_sample <= len(audio_data):
        audio_files[current_audio_id]['data'] = normalize_audio(audio_data[start_sample:end_sample])
        return jsonify({'message': 'Audio trimmed'}), 200
    return jsonify({'error': 'Invalid trim parameters'}), 400

@app.route('/reverse', methods=['POST'])
def reverse_audio():
    if current_audio_id is None or current_audio_id not in audio_files:
        return jsonify({'error': 'No audio selected'}), 400
    audio_files[current_audio_id]['data'] = normalize_audio(audio_files[current_audio_id]['data'][::-1])
    return jsonify({'message': 'Audio reversed'}), 200

@app.route('/speed', methods=['POST'])
def change_speed():
    if current_audio_id is None or current_audio_id not in audio_files:
        return jsonify({'error': 'No audio selected'}), 400
    data = request.get_json()
    speed = float(data['speed'])
    audio_data = audio_files[current_audio_id]['data']
    # Suppress librosa warnings about short signals
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        audio_files[current_audio_id]['data'] = normalize_audio(
            librosa.effects.time_stretch(audio_data.T, rate=speed, n_fft=256).T
        )
    return jsonify({'message': 'Speed changed'}), 200

@app.route('/pitch', methods=['POST'])
def change_pitch():
    if current_audio_id is None or current_audio_id not in audio_files:
        return jsonify({'error': 'No audio selected'}), 400
    data = request.get_json()
    semitones = float(data['semitones'])
    audio_data = audio_files[current_audio_id]['data']
    sample_rate = audio_files[current_audio_id]['sample_rate']
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        audio_files[current_audio_id]['data'] = normalize_audio(
            librosa.effects.pitch_shift(audio_data.T, sr=sample_rate, n_steps=semitones, n_fft=256).T
        )
    return jsonify({'message': 'Pitch changed'}), 200

@app.route('/loop', methods=['POST'])
def loop_audio():
    if current_audio_id is None or current_audio_id not in audio_files:
        return jsonify({'error': 'No audio selected'}), 400
    data = request.get_json()
    count = int(data['count'])
    if count > 0:
        audio_files[current_audio_id]['data'] = normalize_audio(
            np.tile(audio_files[current_audio_id]['data'], (count, 1))
        )
        return jsonify({'message': 'Audio looped'}), 200
    return jsonify({'error': 'Invalid loop count'}), 400

@app.route('/concatenate', methods=['POST'])
def concatenate_audio():
    if current_audio_id is None or current_audio_id not in audio_files:
        return jsonify({'error': 'No audio selected'}), 400
    data = request.get_json()
    second_audio_id = data['second_audio_id']
    if second_audio_id not in audio_files:
        return jsonify({'error': 'Second audio not found'}), 404
    audio_data = audio_files[current_audio_id]['data']
    sample_rate = audio_files[current_audio_id]['sample_rate']
    concat_data = audio_files[second_audio_id]['data']
    concat_rate = audio_files[second_audio_id]['sample_rate']
    if concat_rate != sample_rate:
        concat_data = librosa.resample(concat_data.T, orig_sr=concat_rate, target_sr=sample_rate).T
    audio_files[current_audio_id]['data'] = normalize_audio(np.concatenate((audio_data, concat_data), axis=0))
    return jsonify({'message': 'Audio concatenated'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)