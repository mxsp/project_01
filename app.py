import os
import secrets
import traceback
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
from werkzeug.security import generate_password_hash, check_password_hash #for secure password handling


app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
socketio = SocketIO(app)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # Redirect to login page if not logged in

# User database (REPLACE WITH A REAL DATABASE IN PRODUCTION!)
# This is for demonstration only and is NOT secure for production use.
users = {
    "admin": generate_password_hash("pass") #Hash the password!
}

class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    # In a real application, this would query a database
    return User(user_id) if user_id in users else None


PYTHON_FILES_DIR = 'python_files'
os.makedirs(PYTHON_FILES_DIR, exist_ok=True)

global_namespace = {}
unsafe_globals = {'__builtins__': {}, 'open': None, 'compile': None, 'eval': None, 'exec': None}


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
        print(f"Error getting GPU usage: {e}")
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


def execute_code(code):
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    plt.clf()  # Clear any existing plots
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


def get_matplotlib_image():
    if plt.gcf().axes:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        fig = plt.gcf()
        width, height = fig.get_size_inches() * fig.dpi
        return f"data:image/png;base64,{img_str}", int(width), int(height)
    return None, 0, 0


@socketio.on('run_cell')
@login_required
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


@socketio.on('run_custom_cell')
@login_required
def handle_run_custom_cell(data):
    global global_namespace
    cell_id = data['cell_id']
    code = data['code']
    params = data['params']
    custom_cell_type = data.get('type')

    try:
        if custom_cell_type == 'example1':
            num_layers = int(params.get('layers', 2))
            num_neurons = int(params.get('neurons', 32))
            activation = params.get('activation', 'relu')
            loss = params.get('loss', 'binary_crossentropy')
            optimizer = params.get('optimizer', 'adam')
            epochs = int(params.get('epochs', 10))

            keras_code = generate_keras_model_code(num_layers, num_neurons, activation, loss, optimizer, epochs)
            output_code, error_code, updated_namespace, image_data = execute_code(keras_code)
            global global_namespace
            global_namespace = updated_namespace
            output = ""
            if error_code:
                output = f"\n\nError in generated code:\n{error_code}"
            else:
                output = f"\n\nGenerated code output:\n{output_code}"

        elif custom_cell_type == 'example2':
            x_data = params.get('x_data', [1, 2, 3, 4, 5])
            y_data = params.get('y_data', [2, 4, 1, 3, 5])

            try:
                x_data = ast.literal_eval(x_data)
                y_data = ast.literal_eval(y_data)
            except (ValueError, SyntaxError):
                pass

            if not isinstance(x_data, list) or not isinstance(y_data, list):
                return socketio.emit('cell_output', {'cell_id': cell_id, 'output': "Error: x_data and y_data must be lists.", 'variables': get_variables(), 'imageData': None})

            plt.plot(x_data, y_data)
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.title("Simple Plot")
            image_data = get_matplotlib_image()
            output = ""

        elif custom_cell_type == 'example3':
            csv_filename = params.get('filename')
            if not csv_filename:
                return socketio.emit('cell_output', {'cell_id': cell_id, 'output': "Error: No CSV filename provided.", 'variables': get_variables(), 'imageData': None})

            filepath = os.path.join(PYTHON_FILES_DIR, csv_filename)
            if not os.path.exists(filepath):
                return socketio.emit('cell_output', {'cell_id': cell_id, 'output': f"Error: CSV file '{csv_filename}' not found.", 'variables': get_variables(), 'imageData': None})

            try:
                df = pd.read_csv(filepath)
                output = df.head().to_string()
            except pd.errors.EmptyDataError:
                output = "Error: CSV file is empty."
            except pd.errors.ParserError:
                output = "Error: Could not parse CSV file."
            except Exception as e:
                output = f"An unexpected error occurred: {e}"
            image_data = None

        else:
            raise ValueError(f"Unknown custom cell type: {custom_cell_type}")

        socketio.emit('cell_output', {'cell_id': cell_id, 'output': output, 'variables': get_variables(), 'imageData': image_data, 'keras_code': keras_code if custom_cell_type == 'example1' else ''})

    except Exception as e:
        error_msg = traceback.format_exc()
        socketio.emit('cell_output', {'cell_id': cell_id, 'output': f"Error: {error_msg}", 'variables': get_variables(), 'imageData': None, 'keras_code': ''})


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
    if filename is None:
        filename = 'notebook.py'
    filepath = os.path.join(PYTHON_FILES_DIR, filename)

    try:
        with open(filepath, 'w') as f:
            f.write(notebook_content)
        emit('save_result', {'success': True, 'message': f'Notebook saved as {filename}'})
    except Exception as e:
        emit('save_result', {'success': False, 'message': f'Error saving notebook: {e}'})



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User(username)
        if user and check_password_hash(users.get(username), password): #Check password hash securely
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')
    return render_template('login.html')


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))


if __name__ == '__main__':
    socketio.run(app, debug=True)