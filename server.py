import io
import json
import os
import traceback
from typing import *
import numpy as np

import soundfile as sf

from flask import Flask, make_response, request, send_file
from scipy.io.wavfile import write

from modules.server.model import VoiceServerModel
import subprocess


model: Optional[VoiceServerModel] = None
app = Flask(__name__)

@app.route('/tts', methods=['POST'])
def tts():
    """
    input:
        form:
            text: str
                The text to be read.
            rate: float (optional)
                The speech rate (speed) at which the text should be read.
            volume: float (optional)
                The volume level of the speech.
            pitch: float (optional)
                The pitch level of the speech.
            voice: str (optional)
                The voice model to use for TTS.

    output:
        wavfile

    curl:
        ```
        $ curl -X POST -F "text=Hello, world!" -F "rate=1.0" -F "volume=1.0" -F "pitch=1.0" -F "voice=default_voice" \
            http://localhost:5001/tts > ./output.wav
        ```
    """
    print("TTS request received")
    if request.method == "POST":
        text = request.form.get("text", "")
        if not text:
            return make_response("No text provided", 400)

        rate = request.form.get("rate")
        volume = request.form.get("volume")
        pitch = request.form.get("pitch")
        voice = request.form.get("voice")

        # Path for the audio and subtitle output
        audio_output_path = "output.wav"  # Specify the output file name
        subtitles_output_path = "output.vtt"  # Specify the subtitles output file name
        command = f"edge-tts --text \"{text}\" --write-media {audio_output_path} --write-subtitles {subtitles_output_path} --voice ja-JP-NanamiNeural"
        if rate:
            command += f" --rate {rate}"
        if volume:
            command += f" --volume {volume}"
        if pitch:
            command += f" --pitch {pitch}"
        if voice:
            command += f" --voice {voice}"

        try:
            subprocess.run(command, check=True, shell=True)
            return send_file(audio_output_path, mimetype="audio/wav")
        except subprocess.CalledProcessError as e:
            print("An error occurred while running edge-tts:", e)
            return make_response("Error in processing TTS", 500)
    else:
        return make_response("Use POST method", 400)

@app.route('/ping')
def ping():
    return make_response("server is alive", 200)

@app.route('/convert_sound', methods=['POST'])
def convert_sound():
    """
    input:
        form:
            model_name: str (optional)
                specify the model name
            speaker_id: int (optional)
                default: 0
            transpose: int (optional)
                default: 0
            pitch_extraction_algo: str (optional)
                default: dio
                value: ["dio", "harvest", "mangio-crepe", "crepe"]
            retrieval_feature_ratio: float (optional)
                default: 0
                value: 0. ~ 1.
        input_wav: wav file

    output:
        wavfile

    curl:
        ```
        $ curl -X POST -F "input_wav=@./input.mp3" http://localhost:5001/convert_sound \
            -F "model_name=<YOUR_MODEL_NAME>.pth" \
            >| ./outputs/output.wav
        ```

    """
    print("start")
    if request.method == "POST":
        input_buffer = io.BytesIO(request.files["input_wav"].stream.read())
        audio, sr = sf.read(input_buffer)

        # stereo -> mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        model_name = request.form.get("model_name", "")
        sid = int(request.form.get("speaker_id", 0))
        transpose = int(request.form.get("transpose", 0))
        pitch_extraction_algo = request.form.get("pitch_extraction_algo", "dio")
        if pitch_extraction_algo not in ["dio", "harvest", "mangio-crepe", "crepe"]:
            return make_response("bad pitch extraction algo", 400)
        retrieval_feature_ratio = float(request.form.get("retrieval_feature_ratio", 0.))

        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_script_dir, 'models', 'checkpoints', model_name)

        model = VoiceServerModel(model_path, "")
        out_audio = model(audio, sr, sid, transpose, pitch_extraction_algo, retrieval_feature_ratio)
        output_buffer = io.BytesIO()
        write(output_buffer, rate=model.tgt_sr, data=out_audio)
        output_buffer.seek(0)
        response = make_response(send_file(output_buffer, mimetype="audio/wav"), 200)
        return response
    else:
        return make_response("use post method", 400)

if __name__ == "__main__":
    # app.run()
    app.run(host='0.0.0.0', port=5001)