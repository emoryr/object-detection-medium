import onnx
import torch
from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.datasets import LoadImages
from utils.general import check_img_size

model_path = '' #path to file .pt
images_path = '' #path to direct with images

def get_model_input():
    device = select_device('cpu')
    model = attempt_load('model_path', map_location=device)
    half = device.type != 'cpu'

    ##read image
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(416, s=stride)  # check img_size
    dataset = LoadImages(images_path, img_size=imgsz, stride=stride)
    img_out = None
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        img_out = img
        break;

    return img_out, model

example_input, pytorch_model = get_model_input() # example for the forward pass input
ONNX_PATH= "" #whete should it be saved

#torch_out = pytorch_model(example_input)

torch.onnx.export(
    pytorch_model,
    example_input, 
    ONNX_PATH,
    use_external_data_format=True,
    opset_version=11,
    export_params=True,
    do_constant_folding=False,  # fold constant values for optimization
    input_names=['input'],
    output_names=['output']
)
onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
