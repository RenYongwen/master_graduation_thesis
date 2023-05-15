import torch
import numpy
import random,json
import soundfile
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import wave
app = Flask(__name__)

model = torch.jit.load('model_0027.pt')


def process_audio(file_name):
 # Preprocess image for model
    audio, sr = soundfile.read(file_name)
    length = 200 * 160 + 240
    if audio.shape[0] <= length:
        shortage = length - audio.shape[0]
        audio = numpy.pad(audio, (0, shortage), 'wrap')
    start_frame = numpy.int64(random.random()*(audio.shape[0]-length))
    audio = audio[start_frame:start_frame + length]
    audio = numpy.stack([audio], axis=0)
    return torch.FloatTensor(audio[0])


class_names = ['Arabic', 'Basque', 'Breton', 'Catalan', 'Chinese_China', 'Chinese_Hongkong', 'Chinese_Taiwan', 'Chuvash', 'Czech', 'Dhivehi', 'Dutch', 'English', 'Esperanto', 'Estonian', 'French', 'Frisian', 'Georgian', 'German', 'Greek', 'Hakha_Chin', 'Indonesian', 'Interlingua',
               'Italian', 'Japanese', 'Kabyle', 'Kinyarwanda', 'Kyrgyz', 'Latvian', 'Maltese', 'Mangolian', 'Persian', 'Polish', 'Portuguese', 'Romanian', 'Romansh_Sursilvan', 'Russian', 'Sakha', 'Slovenian', 'Spanish', 'Swedish', 'Tamil', 'Tatar', 'Turkish', 'Ukranian', 'Welsh']


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html', class_probs=[])


@app.route('/predict', methods=['POST', 'GET'])
def predict():
 # Get uploaded image file
    print(request.form)
    file = request.files["audio"]
    with wave.open('temp.wav', 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        while True:
            data = file.read(1024)
            if not data:
                break
            wav_file.writeframes(data)
 # Process image and make prediction
    audio_tensor = process_audio('temp.wav')
    audio_tensor = audio_tensor.unsqueeze(0).cuda()
    output = model(audio_tensor)

 # Get class probabilities
    probabilities = torch.nn.functional.softmax(output, dim=1)
    probabilities = probabilities.cpu().detach().numpy()[0]

 # Sort class probabilities in descending order
    class_probs = list(zip(class_names, probabilities))
    class_probs.sort(key=lambda x: x[1], reverse=True)

 # Render HTML page with prediction results
    json_probs = []
    for c, p in class_probs:
        json_probs.append({'class': c, 'prob': float(p)})
    json_probs = json.dumps(json_probs)

    return json_probs


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
