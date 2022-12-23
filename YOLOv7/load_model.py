import torch
from models.yolo import Model
from utils.torch_utils import ModelEMA, intersect_dicts
import yaml

# Load configuration file
CFG = './cfg/training/yolov7-tiny-hw1.yaml'

# Load weights file
WEIGHTS = './runs/train/exp31/weights/best.pt'

# Number of classes
NC = 8

# Hyperparameter file
HYP = './data/hyp.scratch.hw1.yaml'

# Image size
IMAGE_SIZE = 640

# Intersection over union threshold for non-maximum suppression
iou_thres = 0.1

# Confidence threshold for non-maximum suppression
conf_thres = 0.01

# Load hyperparameters from YAML file
with open(HYP) as f:
    hyp = yaml.load(f, Loader=yaml.SafeLoader)

# Set device to GPU if available, else set to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
model = Model(CFG, ch=3, nc=NC, anchors=None).to(device)

# Load checkpoint
ckpt = torch.load(WEIGHTS, map_location=device)

# Exclude certain keys from checkpoint
exclude = ['anchor']

# Convert checkpoint model to FP32
state_dict = ckpt['model'].float().state_dict()

# Intersect state dict with model state dict, excluding certain keys
state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)

# Load state dict into model
model.load_state_dict(state_dict, strict=False)

# Get number of layers in model
nl = model.model[-1].nl

# Initialize model exponential moving average (EMA)
ema = ModelEMA(model)

# Scale boxes and classes to layers
hyp['box'] *= 3. / nl
hyp['cls'] *= NC / 80. * 3. / nl

# Scale object loss to image size and layers
hyp['obj'] *= (IMAGE_SIZE / 640) ** 2 * 3. / nl

# Set label smoothing to 0
hyp['label_smoothing'] = 0

# Set number of classes in model
model.nc = NC

# Set model intersection over union loss ratio
model.gr = 1.0

# Set model class names
model.names = ['Right_Scissors', 'Left_Scissors', 'Right_Needle_driver', 'Left_Needle_driver', 'Right_Forceps',
               'Left_Forceps', 'Right_Empty', 'Left_Empty']

# Update model EMA
ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
model = ema.ema
