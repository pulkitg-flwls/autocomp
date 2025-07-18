import cv2 as cv
import ptlflow
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter


# Get an optical flow model. As as example, we will use RAFT Small
# with the weights pretrained on the FlyingThings3D dataset
model = ptlflow.get_model('raft_small', ckpt_path='things')

# Load the images
images = [
    cv.imread('./test/dns1.png'),
    cv.imread('./test/dns1.png')
]

# A helper to manage inputs and outputs of the model
io_adapter = IOAdapter(model, images[0].shape[:2])

# inputs is a dict {'images': torch.Tensor}
# The tensor is 5D with a shape BNCHW. In this case, it will have the shape:
# (1, 2, 3, H, W)
inputs = io_adapter.prepare_inputs(images)

# Forward the inputs through the model
predictions = model(inputs)

# The output is a dict with possibly several keys,
# but it should always store the optical flow prediction in a key called 'flows'.
flows = predictions['flows']

# flows will be a 5D tensor BNCHW.
# This example should print a shape (1, 1, 2, H, W).
print(flows.shape)

# Create an RGB representation of the flow to show it on the screen
flow_rgb = flow_utils.flow_to_rgb(flows)
# Make it a numpy array with HWC shape
flow_rgb = flow_rgb[0, 0].permute(1, 2, 0)
flow_rgb_npy = flow_rgb.detach().cpu().numpy()
# OpenCV uses BGR format
flow_bgr_npy = cv.cvtColor(flow_rgb_npy, cv.COLOR_RGB2BGR)
cv.imwrite('flow_bgr.png',(flow_bgr_npy*255).astype('uint8'))