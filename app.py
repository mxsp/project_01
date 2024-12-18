import os
import secrets
import traceback
import threading
import time
from io import StringIO
from flask import Flask, render_template, request, jsonify, send_from_directory, flash, redirect, url_for
from flask_socketio import SocketIO, emit
import ast
import sys
import psutil
import subprocess
import json
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from GPUtil import getGPUs
from flask_cors import CORS
import shutil
import tensorflow as tf
import numpy as np
import pandas as pd
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import re
import logging
from logging.handlers import RotatingFileHandler
import gevent


app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging
log_file = 'app.log'
log_level = logging.INFO

handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
handler.setFormatter(formatter)

app.logger.addHandler(handler)
app.logger.setLevel(log_level)

socketio = SocketIO(app, async_mode='gevent')

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User database (REPLACE WITH A REAL DATABASE IN PRODUCTION!)
users = {
    "admin": generate_password_hash("pass")
}

class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id) if user_id in users else None

PYTHON_FILES_DIR = 'python_files'
os.makedirs(PYTHON_FILES_DIR, exist_ok=True)

global_namespace = {}

# Track the currently active file
current_file = None

# Function to tail the log file and emit updates
def tail_log():
    while True:
        try:
            with open(log_file, 'r') as f:
                f.seek(0, 2)
                while True:
                    line = f.readline()
                    if not line:
                        time.sleep(0.1)
                        continue
                    socketio.emit('terminal_output', {'line': line.strip()})
        except FileNotFoundError:
            app.logger.warning(f"Log file '{log_file}' not found. Retrying...")
            time.sleep(1)

# Start the log tailer in a separate thread
log_thread = threading.Thread(target=tail_log, daemon=True)
log_thread.start()


@app.route('/')
@login_required
def index():
    files = [f for f in os.listdir(PYTHON_FILES_DIR) if f.endswith('.py')]
    uploaded_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER'])]
    return render_template('index.html', files=files, uploaded_files=uploaded_files)

@app.route('/get_files')
@login_required
def get_files():
    files = [f for f in os.listdir(PYTHON_FILES_DIR) if f.endswith('.py') or f.endswith('.csv')]
    return jsonify({'files': files})

@app.route('/get_file_content', methods=['POST'])
@login_required
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
@login_required
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
@login_required
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
@login_required
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
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
@login_required
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
@login_required
def download_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/delete_upload/<filename>', methods=['POST'])
@login_required
def delete_upload(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    try:
        os.remove(filepath)
        return jsonify({'message': 'File deleted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/rename_file', methods=['POST'])
@login_required
def rename_file():
    data = request.json
    old_filename = data.get('oldFilename')
    new_filename = data.get('newFilename')

    if not old_filename or not new_filename:
        return jsonify({'error': 'Old and new filenames are required'}), 400

    old_filepath = os.path.join(app.config['UPLOAD_FOLDER'], old_filename)
    new_filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)

    if not os.path.exists(old_filepath):
        return jsonify({'error': f'File "{old_filename}" not found'}), 404

    try:
        os.rename(old_filepath, new_filepath)
        return jsonify({'message': f'File renamed successfully to "{new_filename}"'})
    except OSError as e:
        return jsonify({'error': f'Error renaming file: {e}'}), 500
    except Exception as e:
        app.logger.exception(f"Unexpected error renaming file: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500


@app.route('/hardware_usage')
@login_required
def hardware_usage():
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    gpu_data = []
    try:
        gpus = getGPUs()
        for gpu in gpus:
            gpu_data.append({'id': gpu.id, 'load': gpu.load * 100, 'memoryUtil': gpu.memoryUtil})
    except Exception as e:
        app.logger.exception(f"Error getting GPU usage: {e}")
    disk_usage = psutil.disk_usage('/').percent
    return jsonify({'cpu': cpu_usage, 'ram': ram_usage, 'gpus': gpu_data, 'disk': disk_usage})

@app.route('/gpu_usage')
@login_required
def gpu_usage():
    try:
        gpus = getGPUs()
        gpu_data = [{'id': gpu.id, 'load': gpu.load, 'memoryUtil': gpu.memoryUtil} for gpu in gpus]
        return jsonify({'gpus': gpu_data})
    except Exception as e:
        return jsonify({'error': f'Error getting GPU usage: {str(e)}'}), 500

def get_matplotlib_image():
    if plt.gcf().axes:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        return f"data:image/png;base64,{img_str}"
    return None

def execute_code(code, cell_id):
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    plt.clf()
    try:
        exec(code, global_namespace)
        output = sys.stdout.getvalue().strip()
        app.logger.info(f"Cell Output ({cell_id}):\n{output}")  # Log to file
        image_data = get_matplotlib_image()
        socketio.emit('cell_output', {'cell_id': cell_id, 'output': output, 'imageData': image_data})
        return "", None, global_namespace, image_data
    except Exception as e:
        error_msg = traceback.format_exc()
        app.logger.error(f"Cell Error ({cell_id}):\n{error_msg}")  # Log error to file
        socketio.emit('cell_output', {'cell_id': cell_id, 'output': error_msg, 'imageData': None})
        return "", error_msg, global_namespace, None
    finally:
        sys.stdout = old_stdout


def generate_keras_model_code(num_layers, num_neurons, activation, loss, optimizer, epochs):
    code = f"""
import tensorflow as tf
import numpy as np
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

model.summary()
"""
    return code

def execute_custom_cell(data):
    code = data['code']
    params = data['params']
    custom_cell_type = data.get('type')
    plt.clf()
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        if custom_cell_type == 'example1':
            num_layers = int(params.get('layers', 2))
            num_neurons = int(params.get('neurons', 32))
            activation = params.get('activation', 'relu')
            loss = params.get('loss', 'binary_crossentropy')
            optimizer = params.get('optimizer', 'adam')
            epochs = int(params.get('epochs', 10))

            keras_code = generate_keras_model_code(num_layers, num_neurons, activation, loss, optimizer, epochs)
            exec(keras_code, global_namespace)
            output = sys.stdout.getvalue().strip()
            app.logger.info(f"Custom Cell Output ({custom_cell_type}-{data.get('cell_id','')}):\n{output}")
            image_data = get_matplotlib_image()
            response = {'output': output, 'imageData': image_data, 'codeToExecute': keras_code}

        elif custom_cell_type == 'example2':
            x_data = params.get('x_data', [1, 2, 3, 4, 5])
            y_data = params.get('y_data', [2, 4, 1, 3, 5])
            try:
                x_data = ast.literal_eval(x_data)
                y_data = ast.literal_eval(y_data)
            except (ValueError, SyntaxError) as e:
                output = f"Error parsing data: {e}"
                image_data = None
                response = {'output': output, 'imageData': image_data, 'codeToExecute': ''}
                return response, None

            code_to_execute = f"""
import matplotlib.pyplot as plt
plt.plot({x_data}, {y_data})
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Simple Plot")
plt.show()
"""
            exec(code_to_execute, global_namespace)
            output = sys.stdout.getvalue().strip()
            app.logger.info(f"Custom Cell Output ({custom_cell_type}-{data.get('cell_id','')}):\n{output}")
            image_data = get_matplotlib_image()
            response = {'output': output, 'imageData': image_data, 'codeToExecute': code_to_execute}

        elif custom_cell_type == 'example3':
            csv_filename = params.get('filename')
            filepath = os.path.join(PYTHON_FILES_DIR, csv_filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"CSV file '{csv_filename}' not found.")

            code_to_execute = f"""
import pandas as pd
df = pd.read_csv('{filepath}')
output_string = df.head().to_string()
print(output_string)
"""
            exec(code_to_execute, global_namespace)
            output = sys.stdout.getvalue().strip()
            app.logger.info(f"Custom Cell Output ({custom_cell_type}-{data.get('cell_id','')}):\n{output}")
            image_data = None
            response = {'output': output, 'imageData': image_data, 'codeToExecute': code_to_execute}

        else:
            raise ValueError(f"Unknown custom cell type: {custom_cell_type}")
        return response, None

    except FileNotFoundError as e:
        app.logger.error(f"Custom Cell Error ({custom_cell_type}-{data.get('cell_id','')}):\n{e}")
        return None, f"Error: {e}"
    except Exception as e:
        app.logger.exception(f"Custom Cell Error ({custom_cell_type}-{data.get('cell_id','')}):\n{e}")
        return None, traceback.format_exc()
    finally:
        sys.stdout = old_stdout







# ... (rest of your routes, SocketIO handlers, etc.) ...
@app.route('/save_cell_order', methods=['POST'])
@login_required
def save_cell_order():
    data = request.json
    cell_ids = data.get('cellIds')
    filename = data.get('filename')

    if not filename or not cell_ids:
        return jsonify({'error': 'Filename and cell IDs are required'}), 400

    filepath = os.path.join(PYTHON_FILES_DIR, filename)
    try:
        with open(filepath, 'r') as f:
            existing_content = f.read()

        new_content = re.sub(r'^# Cell Order:.*$', f'# Cell Order: {", ".join(cell_ids)}', existing_content, flags=re.MULTILINE)

        with open(filepath, 'w') as f:
            f.write(new_content)

        return jsonify({'message': 'Cell order saved successfully'})
    except Exception as e:
        app.logger.exception(f"Error saving cell order: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/load_cell_order', methods=['POST'])
@login_required
def load_cell_order():
    filename = request.json.get('filename')
    if not filename:
        return jsonify({'error': 'Filename not provided'}), 400

    filepath = os.path.join(PYTHON_FILES_DIR, filename)
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith("# Cell Order:"):
                    cell_ids = [x.strip() for x in line[13:].split(',')]
                    return jsonify({'cell_ids': cell_ids})
                    break
            return jsonify({'cell_ids': []})

    except FileNotFoundError:
        return jsonify({'error': f'File "{filename}" not found.'}), 404
    except Exception as e:
        app.logger.exception(f"Error loading cell order: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and check_password_hash(users[username], password):
            login_user(User(username))
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@socketio.on('run_cell')
@login_required
def handle_run_cell(data):
    cell_id = data['cell_id']
    cell_content = data['cell_content']
    execute_code(cell_content, cell_id)

@socketio.on('run_custom_cell')
@login_required
def handle_run_custom_cell(data):
    cell_id = data.get('cell_id')
    if not cell_id:
        return socketio.emit('cell_output', {'cell_id': cell_id, 'output': 'Missing cell_id'}), 400
    try:
        response, error = execute_custom_cell(data)
        if error:
            socketio.emit('cell_output', {'cell_id': cell_id, 'output': error})
        else:
            socketio.emit('cell_output', {'cell_id': cell_id, 'output': response['output'], 'imageData': response['imageData'], 'codeToExecute': response['codeToExecute']})

    except Exception as e:
        socketio.emit('cell_output', {'cell_id': cell_id, 'output': f'Error: {str(e)}'})


@socketio.on('reset_kernel')
@login_required
def handle_reset_kernel():
    global global_namespace
    global_namespace = {}
    emit('kernel_reset', {'variables': get_variables()})

@socketio.on('get_variables')
@login_required
def handle_get_variables():
    emit('variables_update', {'variables': get_variables()})

def get_variables():
    safe_variables = {k: v for k, v in global_namespace.items() if not k.startswith('_')}
    return json.dumps(safe_variables, default=str)

@socketio.on('pip_install')
@login_required
def handle_pip_install(data):
    package_name = data['packageName']
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
        emit('pip_install_result', {'success': True, 'message': f'{package_name} installed successfully.'})
    except subprocess.CalledProcessError as e:
        emit('pip_install_result', {'success': False, 'message': f'Error installing {package_name}: {e}'})
    except Exception as e:
        emit('pip_install_result', {'success': False, 'message': f'An unexpected error occurred: {e}'})

@socketio.on('save_notebook')
@login_required
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
    socketio.run(app, debug=False)
