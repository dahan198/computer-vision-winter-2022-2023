import torch
from yolov7.models.yolo import Model
from yolov7.utils.torch_utils import ModelEMA, intersect_dicts
import yaml

# Number of classes
NC = 8

# Image size
IMAGE_SIZE = 640

# Intersection over union threshold for non-maximum suppression
iou_thres = 0.1

# Confidence threshold for non-maximum suppression
conf_thres = 0.01

HYP = './yolov7/data/hyp.scratch.exp1.yaml'


def load_model(cfg='./yolov7/cfg/training/yolov7-tiny-exp1.yaml',
               weights='./yolov7/runs/train/exp/weights/best.pt'):
    """"
        Loads a trained YOLOv7 model and returns initialized model.
        
    Args:
        cfg (str): the path to the model configuration file in YAML format. It specifies the structure of the model.
        weights (str): the path to the file containing the weights of the trained model.
        
    Returns:
        nn.Module: initialized model. 
    """""

    # Load hyperparameters from YAML file
    with open(HYP) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)

    # Set device to GPU if available, else set to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = Model(cfg, ch=3, nc=NC, anchors=None).to(device)

    # Load checkpoint
    ckpt = torch.load(weights, map_location=device)

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
    return model
