from flask import Flask, jsonify, request, render_template, send_file
from flask_cors import CORS
import os
from backend.train_manager import TrainManager

app = Flask(__name__)
CORS(app)  # Enable CORS for Vue frontend

train_manager = TrainManager()
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs")


def _is_safe_path(path):
    abs_path = os.path.abspath(path)
    return abs_path.startswith(os.path.abspath(OUTPUT_ROOT))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify(train_manager.get_status())

@app.route('/api/train', methods=['POST'])
def start_training():
    config = request.json
    success, message = train_manager.start_training(config)
    if success:
        return jsonify({"status": "success", "message": message})
    else:
        return jsonify({"status": "error", "message": message}), 400

@app.route('/api/stop', methods=['POST'])
def stop_training():
    success, message = train_manager.stop_training()
    return jsonify({"status": "success" if success else "error", "message": message})

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    # List available processed datasets
    data_dir = 'data/processed'
    if not os.path.exists(data_dir):
        return jsonify([])
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    return jsonify(files)


@app.route('/api/download', methods=['GET'])
def download_file():
    target_path = request.args.get('path', '')
    if not target_path:
        return jsonify({"status": "error", "message": "Missing path parameter"}), 400

    abs_path = os.path.abspath(target_path)
    if not _is_safe_path(abs_path):
        return jsonify({"status": "error", "message": "Path is not allowed"}), 403
    if not os.path.exists(abs_path):
        return jsonify({"status": "error", "message": "File not found"}), 404

    return send_file(abs_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
