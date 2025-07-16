

# Project specific imports
import numpy as np

# Flawless components
from flwls_optical_flow import FlowSystem, FlowSystemConfig
from flwls_optical_flow.temp_common_io import FlowSystemInput


def test_component_high_level() -> None:
    # Instantiate our component and ensure that we can
    # instantiate our component without having to supply
    # mandatory additional parameters
    optical_flow = FlowSystem(FlowSystemConfig(
        batch_size = 1,
        model_name ='Raft',
        flow_direction = "both",
        device = "cuda",
    ))

    # Prepare test data
    image_array = np.random.default_rng().integers(low=0, high=256, size=(200, 200, 3), dtype=np.uint8)
    value = [FlowSystemInput(image=image_array[:-1, :-1]), FlowSystemInput(image=image_array[:-1, :-1])]

    # Execute primary high-level method
    output = list(optical_flow(value))[0]

    # Validate output
    # assert len(output) == 1
    output = output.to_dict()
    forward = output['forward_flow'].transpose(2,1,0)
    
    print(forward.max(),forward.min())
    return 

if __name__=="__main__":
    test_component_high_level()