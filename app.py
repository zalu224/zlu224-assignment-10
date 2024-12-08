from flask import Flask, render_template, request, jsonify
import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from PIL import Image
import open_clip
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load CLIP model and preprocessing
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Load image embeddings database
df = pd.read_pickle('image_embeddings.pickle')
image_paths = df.iloc[:, 0].values
embeddings = np.vstack(df.iloc[:, 1].values)

def allowed_file(filename):
    """Check if the file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def safe_filename(filename):
    """Generate a safe filename while preserving extension"""
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    return f"{uuid.uuid4().hex}.{ext}"

def text_search(text_query):
    # Encode and normalize text query
    text = tokenizer([text_query])
    text_embedding = F.normalize(model.encode_text(text))
    
    # Calculate similarities
    similarities = []
    for embedding in embeddings:
        img_embedding = torch.from_numpy(embedding)
        img_embedding_normalized = F.normalize(img_embedding.unsqueeze(0))
        similarity = F.cosine_similarity(text_embedding, img_embedding_normalized)
        similarities.append(similarity.item())
    
    # Get top 5 results
    top_indices = np.argsort(similarities)[-5:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            'image_path': image_paths[idx],
            'similarity': similarities[idx]
        })
    return results

def image_search(image_path):
    # Load and preprocess image
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    image_embedding = F.normalize(model.encode_image(image))
    
    # Calculate similarities
    similarities = []
    for embedding in embeddings:
        img_embedding = torch.from_numpy(embedding)
        img_embedding_normalized = F.normalize(img_embedding.unsqueeze(0))
        similarity = F.cosine_similarity(image_embedding, img_embedding_normalized)
        similarities.append(similarity.item())
    
    # Get top 5 results
    top_indices = np.argsort(similarities)[-5:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            'image_path': image_paths[idx],
            'similarity': similarities[idx]
        })
    return results

def hybrid_search(image_path, text_query, weight):
    # Encode and normalize text query
    text = tokenizer([text_query])
    text_embedding = F.normalize(model.encode_text(text))
    
    # Encode and normalize image query
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    image_embedding = F.normalize(model.encode_image(image))
    
    # Combine embeddings with weight
    query = F.normalize(weight * text_embedding + (1.0 - weight) * image_embedding)
    
    # Calculate similarities
    similarities = []
    for embedding in embeddings:
        img_embedding = torch.from_numpy(embedding)
        img_embedding_normalized = F.normalize(img_embedding.unsqueeze(0))
        similarity = F.cosine_similarity(query, img_embedding_normalized)
        similarities.append(similarity.item())
    
    # Get top 5 results
    top_indices = np.argsort(similarities)[-5:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            'image_path': image_paths[idx],
            'similarity': similarities[idx]
        })
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    search_type = request.form.get('search_type')
    results = []
    
    if search_type == 'text':
        text_query = request.form.get('text_query')
        if text_query:
            results = text_search(text_query)
            
    elif search_type == 'image':
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'})
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'})
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'})
            
        filename = safe_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            results = image_search(filepath)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)  # Clean up uploaded file
        
    elif search_type == 'hybrid':
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'})
        file = request.files['image']
        text_query = request.form.get('text_query')
        weight = float(request.form.get('weight', 0.5))
        
        if file.filename == '':
            return jsonify({'error': 'No image selected'})
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'})
            
        filename = safe_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            results = hybrid_search(filepath, text_query, weight)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)  # Clean up uploaded file

    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True)