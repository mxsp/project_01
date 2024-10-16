import os
import secrets
import traceback
from io import StringIO
from flask import Flask, render_template, request, jsonify
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

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
socketio = SocketIO(app)

PYTHON_FILES_DIR = 'python_files'
os.makedirs(PYTHON_FILES_DIR, exist_ok=True)

global_namespace = {}
unsafe_globals = {'__builtins__': {}, 'open': None, 'compile': None, 'eval': None, 'exec': None}

@app.route('/')
def index():
    files = [f for f in os.listdir(PYTHON_FILES_DIR) if f.endswith('.py')]
    return render_template('index.html', files=files)

@app.route('/get_files')
def get_files():
    files = [f for f in os.listdir(PYTHON_FILES_DIR) if f.endswith('.py')]
    return jsonify({'files': files})

@app.route('/get_file_content', methods=['POST'])
def get_file_content():
    filename = request.json['filename']
    file_path = os.path.join(PYTHON_FILES_DIR, filename)
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return jsonify({'content': content})
    except FileNotFoundError:
        return jsonify({'error': f'File "{filename}" not found.'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/save_file', methods=['POST'])
def save_file():
    filename = request.json['filename']
    content = request.json['content']
    file_path = os.path.join(PYTHON_FILES_DIR, filename)
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        return jsonify({'message': 'File saved successfully'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/create_file', methods=['POST'])
def create_file():
    filename = request.json['filename']
    file_path = os.path.join(PYTHON_FILES_DIR, filename)
    if not os.path.exists(file_path):
        try:
            with open(file_path, 'w') as file:
                file.write('# New Python file\n')
            return jsonify({'message': f'{filename} created successfully'})
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({'error': f'{filename} already exists'})

@app.route('/delete_file', methods=['POST'])
def delete_file():
    filename = request.json['filename']
    file_path = os.path.join(PYTHON_FILES_DIR, filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return jsonify({'message': f'{filename} deleted successfully'})
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({'error': f'{filename} does not exist'})

@app.route('/hardware_usage')
def hardware_usage():
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    return jsonify({'cpu': cpu_usage, 'ram': ram_usage, 'gpu': 0})

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
        allowed_packages = ['matplotlib', 'numpy', 'pandas']
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

@socketio.on('toggle_gpu_monitor')
def handle_toggle_gpu_monitor(data):
    emit('gpu_monitor_update', {'show': data['show']})

if __name__ == '__main__':
    socketio.run(app, debug=True)
