import os
import secrets
import traceback
from io import StringIO
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import ast
import sys
import psutil
import subprocess
import json
import base64
import matplotlib.pyplot as plt
import io
from GPUtil import getGPUs
from flask_cors import CORS
import shutil
import tensorflow as tf
import numpy as np

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
socketio = SocketIO(app)

PYTHON_FILES_DIR = 'python_files'
os.makedirs(PYTHON_FILES_DIR, exist_ok=True)

global_namespace = {}
unsafe_globals = {'__builtins__': {}, 'open': None, 'compile': None, 'eval': None, 'exec': None}

@app.route('/')
def index():
    files = [f for f in os.listdir(PYTHON_FILES_DIR) if f.endswith('.py')]
    uploaded_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER'])]
    return render_template('index.html', files=files, uploaded_files=uploaded_files)

@app.route('/get_files')
def get_files():
    files = [f for f in os.listdir(PYTHON_FILES_DIR) if f.endswith('.py')]
    return jsonify({'files': files})

@app.route('/get_file_content', methods=['POST'])
def get_file_content():
    filename = request.json.get('filename')
    if not filename:
        return jsonify({'error': 'Filename not provided'}), 400

    file_path = os.path.join(PYTHON_FILES_DIR, filename)
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return jsonify({'content': content})
    except FileNotFoundError:
        return jsonify({'error': f'File "{filename}" not found.'}), 404
    except Exception as e:
        app.logger.exception(f"Error getting file content: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/save_file', methods=['POST'])
def save_file():
    data = request.json
    filename = data.get('filename')
    content = data.get('content')
    if not filename or not content:
        return jsonify({'error': 'Filename and content are required'}), 400

    file_path = os.path.join(PYTHON_FILES_DIR, filename)
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        return jsonify({'message': 'File saved successfully'})
    except Exception as e:
        app.logger.exception(f"Error saving file: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/create_file', methods=['POST'])
def create_file():
    filename = request.json.get('filename')
    if not filename:
        return jsonify({'error': 'Filename not provided'}), 400

    file_path = os.path.join(PYTHON_FILES_DIR, filename)
    if os.path.exists(file_path):
        return jsonify({'error': f'{filename} already exists'}), 409
    try:
        with open(file_path, 'w') as file:
            file.write('# New Python file\n')
        return jsonify({'message': f'{filename} created successfully'})
    except Exception as e:
        app.logger.exception(f"Error creating file: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/delete_file', methods=['POST'])
def delete_file():
    filename = request.json.get('filename')
    if not filename:
        return jsonify({'error': 'Filename not provided'}), 400

    file_path = os.path.join(PYTHON_FILES_DIR, filename)
    if not os.path.exists(file_path):
        return jsonify({'error': f'{filename} does not exist'}), 404
    try:
        os.remove(file_path)
        return jsonify({'message': f'{filename} deleted successfully'})
    except Exception as e:
        app.logger.exception(f"Error deleting file: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secrets.token_hex(8) + os.path.splitext(file.filename)[1]
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({'filename': filename, 'message': 'File uploaded successfully'})

@app.route('/download/<filename>')
def download_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/delete_upload/<filename>', methods=['POST'])
def delete_upload(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    try:
        os.remove(filepath)
        return jsonify({'message': 'File deleted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/hardware_usage')
def hardware_usage():
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    gpu_data = []
    try:
        gpus = getGPUs()
        for gpu in gpus:
            gpu_data.append({'id': gpu.id, 'load': gpu.load*100, 'memoryUtil': gpu.memoryUtil})
    except Exception as e:
        print(f"Error getting GPU usage: {e}")
    disk_usage = psutil.disk_usage('/').percent
    return jsonify({'cpu': cpu_usage, 'ram': ram_usage, 'gpus': gpu_data, 'disk': disk_usage})

@app.route('/gpu_usage')
def gpu_usage():
    try:
        gpus = getGPUs()
        gpu_data = [{'id': gpu.id, 'load': gpu.load, 'memoryUtil': gpu.memoryUtil} for gpu in gpus]
        return jsonify({'gpus': gpu_data})
    except Exception as e:
        return jsonify({'error': f'Error getting GPU usage: {str(e)}'}), 500

def execute_code(code):
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    old_plt_backend = plt.get_backend()
    plt.switch_backend('Agg')
    try:
        exec(code, global_namespace, unsafe_globals)
        output = redirected_output.getvalue().strip()
        image_data = get_matplotlib_image()
        return output, None, global_namespace, image_data
    except Exception as e:
        error_msg = traceback.format_exc()
        return "", error_msg, global_namespace, None
    finally:
        sys.stdout = old_stdout
        plt.switch_backend(old_plt_backend)

def get_matplotlib_image():
    if plt.gcf().get_axes():
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        return f"data:image/png;base64,{img_str}"
    return None

@socketio.on('run_cell')
def handle_run_cell(data):
    cell_id = data['cell_id']
    cell_content = data['cell_content']
    timeout = int(data.get('timeout', 10))

    output, error, updated_namespace, image_data = execute_code(cell_content)
    global global_namespace
    global_namespace = updated_namespace

    result = output if error is None else error
    socketio.emit('cell_output', {'cell_id': cell_id, 'output': result, 'variables': get_variables(), 'imageData': image_data})

def generate_keras_model_code(num_layers, num_neurons, activation, loss, optimizer, epochs):
    code = f"""
import tensorflow as tf
import numpy as np
import io
import json
num_layers = {num_layers}
num_neurons = {num_neurons}
activation = '{activation}'
loss = '{loss}'
optimizer = '{optimizer}'
epochs = {epochs}

# Generate synthetic data
num_samples = 100
input_dim = 10
output_dim = 1
x_train = np.random.rand(num_samples, input_dim)
y_train = np.random.randint(0, 2, (num_samples, output_dim))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(num_neurons, activation=activation, input_shape=(input_dim,)),
    * [tf.keras.layers.Dense(num_neurons, activation=activation) for _ in range(num_layers - 1)],
    tf.keras.layers.Dense(output_dim, activation='sigmoid')
])

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=epochs, verbose=0)


output = model.summary
print(output)

"""
    return code
    

@socketio.on('run_custom_cell')
def handle_run_custom_cell(data):
    cell_id = data['cell_id']
    code = data['code']
    params = data['params']

    try:
        num_layers = int(params.get('layers', 2))
        num_neurons = int(params.get('neurons', 32))
        activation = params.get('activation', 'relu')
        loss = params.get('loss', 'binary_crossentropy')
        optimizer = params.get('optimizer', 'adam')
        epochs = int(params.get('epochs', 10))

        if num_layers < 1 or num_neurons < 1 or epochs < 1:
            raise ValueError("Layers, neurons, and epochs must be positive integers.")


        keras_code = generate_keras_model_code(num_layers, num_neurons, activation, loss, optimizer, epochs)

        output_code, error_code, updated_namespace, image_data = execute_code(keras_code)
        global global_namespace
        global_namespace = updated_namespace

        output = ""
        if error_code:
            output = f"\n\nError in generated code:\n{error_code}"
        else:
            output = f"\n\nGenerated code output:\n{output_code}"

        socketio.emit('cell_output', {'cell_id': cell_id, 'output': output, 'variables': get_variables(), 'imageData': image_data, 'keras_code': keras_code})

    except Exception as e:
        error_msg = traceback.format_exc()
        socketio.emit('cell_output', {'cell_id': cell_id, 'output': error_msg, 'variables': get_variables(), 'imageData': None, 'keras_code': ''})


@socketio.on('reset_kernel')
def handle_reset_kernel():
    global global_namespace
    global_namespace = {}
    emit('kernel_reset', {'variables': get_variables()})

@socketio.on('get_variables')
def handle_get_variables():
    emit('variables_update', {'variables': get_variables()})

def get_variables():
    safe_variables = {k: v for k, v in global_namespace.items() if not k.startswith('_')}
    return json.dumps(safe_variables, default=str)

@socketio.on('pip_install')
def handle_pip_install(data):
    package_name = data['packageName']
    try:
        allowed_packages = ['matplotlib', 'numpy', 'pandas', 'tensorflow', 'keras'] #add more packages as needed
        if package_name in allowed_packages:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
            emit('pip_install_result', {'success': True, 'message': f'{package_name} installed successfully.'})
        else:
            emit('pip_install_result', {'success': False, 'message': f'Installation of {package_name} is not allowed.'})
    except subprocess.CalledProcessError as e:
        emit('pip_install_result', {'success': False, 'message': f'Error installing {package_name}: {e}'})
    except Exception as e:
        emit('pip_install_result', {'success': False, 'message': f'An unexpected error occurred: {e}'})

@socketio.on('save_notebook')
def handle_save_notebook(data):
    notebook_content = data['notebookContent']
    filename = data.get('filename', 'notebook.py')
    filepath = os.path.join(PYTHON_FILES_DIR, filename)

    try:
        with open(filepath, 'w') as f:
            f.write(notebook_content)
        emit('save_result', {'success': True, 'message': f'Notebook saved as {filename}'})
    except Exception as e:
        emit('save_result', {'success': False, 'message': f'Error saving notebook: {e}'})

if __name__ == '__main__':
    socketio.run(app, debug=True)