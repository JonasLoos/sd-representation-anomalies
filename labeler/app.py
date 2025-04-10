from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import os
import json

app = Flask(__name__)

IMAGE_DIR = "images"

def get_image_files():
    return [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

def get_unlabeled_images():
    all_images = get_image_files()
    return [img for img in all_images if not os.path.exists(os.path.join(IMAGE_DIR, f"{os.path.splitext(img)[0]}.json"))]

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)

@app.route('/api/images', methods=['GET'])
def get_images():
    return jsonify(get_image_files())

@app.route('/api/unlabeled', methods=['GET'])
def get_unlabeled():
    return jsonify(get_unlabeled_images())

@app.route('/api/labels/<image_name>', methods=['GET', 'POST', 'DELETE'])
def handle_labels(image_name: str) -> Response:
    json_path = os.path.join(IMAGE_DIR, f"{os.path.splitext(image_name)[0]}.json")
    
    if request.method == 'GET':
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                return jsonify(json.load(f))
        else:
            return jsonify([])
    
    elif request.method == 'POST':
        labels = request.json
        with open(json_path, 'w') as f:
            json.dump(labels, f)
        return jsonify({"status": "success"})
    
    elif request.method == 'DELETE':
        if os.path.exists(json_path):
            os.remove(json_path)
            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "file not found"}), 404

@app.route('/api/labels', methods=['GET'])
def get_all_labels() -> Response:
    labels = {}
    for filename in os.listdir(IMAGE_DIR):
        if filename.lower().endswith('.json'):
            image_name = os.path.splitext(filename)[0]
            with open(os.path.join(IMAGE_DIR, filename), 'r') as f:
                labels[image_name] = json.load(f)
    return jsonify(labels)

if __name__ == '__main__':
    app.run(debug=True, port=5001)

