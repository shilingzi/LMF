from flask import Flask, request, render_template, send_file
import os
from PIL import Image
import torch
from torchvision import transforms
import models
from utils import make_coord
from test import batched_predict
import io

app = Flask(__name__)

# 配置上传文件存储路径
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# 确保上传和输出目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 初始化模型
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'save/edsr-b_lm-lmlte/epoch-best.pth'
model_spec = torch.load(model_path, map_location=torch.device('cpu'))['model']
model = models.make(model_spec, load_sd=True).to(DEVICE)
model.eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(input_path, scale=4):
    img = transforms.ToTensor()(Image.open(input_path).convert('RGB')).to(DEVICE)
    
    h = int(img.shape[-2] * scale)
    w = int(img.shape[-1] * scale)
    scale = h / img.shape[-2]
    coord = make_coord((h, w)).to(DEVICE)
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    
    cell_factor = max(scale/4, 1)
    with torch.no_grad():
        pred = model(((img - 0.5) / 0.5).unsqueeze(0),
                     coord.unsqueeze(0), cell_factor * cell.unsqueeze(0))[0]
    
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    return transforms.ToPILImage()(pred)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file and allowed_file(file.filename):
        # 保存上传的文件
        filename = file.filename
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # 处理图片
        scale = int(request.form.get('scale', 4))
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'sr_{filename}')
        result = process_image(input_path, scale)
        result.save(output_path)
        
        # 返回处理后的图片
        return send_file(output_path, mimetype='image/png')
    
    return 'Invalid file type', 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 