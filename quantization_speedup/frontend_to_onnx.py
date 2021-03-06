# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import onnx
import onnx.numpy_helper
import struct
"""
The main function of this page is to convert pytorch model to onnx model.
Convertion from pytorch model to onnx model is primary so that a critical
problem is caused that Layer name of pytorch model fail to convert to onnx
layer name directly. To solve it, we wrap pytorch model in new wrapper which
multiply bits number and input before computation of each op. Only in this
way can onnx model get bits number of corresponded layer.
"""

class LayernameModuleWrapper(torch.nn.Module):
    def __init__(self, module, module_bits) -> None:
        """
        Parameters
        ----------
        module : torch.nn.Module
            Layer module of pytorch model
        module_bits : int
            Bits width setting for module
        """
        super().__init__()
        self.module = module
        self.module_bits = module_bits

    def forward(self, inputs):
        inputs = inputs*self.module_bits
        inputs = self.module(inputs)
        return inputs

def _setattr(model, name, module):
    """
    Parameters
    ----------
    model : pytorch model
        The model to speed up by quantization
    name : str
        name of pytorch module
    module : torch.nn.Module
        Layer module of pytorch model
    """
    name_list = name.split(".")
    for name in name_list[:-1]:
        model = getattr(model, name)
    setattr(model, name_list[-1], module)

def unwrapper(model_onnx, index2name, config):
    """
    Fill onnx config and remove wrapper node in onnx

    Parameters
    ----------
    model_onnx : onnx model
        Onnx model which is converted from pytorch model
    index2name : dict
        Dictionary of layer index and name
    config : dict
        Config recording name of layers and calibration parameters

    Returns
    -------
    onnx model
        Onnx model which is converted from pytorch model
    dict
        The configuration of onnx model layers and calibration parameters
    """
    # Support Gemm, Conv, Relu, Clip(Relu6) and Maxpool
    support_op = ['Gemm', 'Conv', 'Relu', 'Clip', 'MaxP']
    idx = 0
    onnx_config = {}
    while idx < len(model_onnx.graph.node):
        nd = model_onnx.graph.node[idx]
        if nd.name[0:4] in support_op and  idx > 1:
            # Grad constant node and multiply node
            const_nd = model_onnx.graph.node[idx-2]
            mul_nd = model_onnx.graph.node[idx-1]
            # Get index number which is transferred by constant node
            index = int(onnx.numpy_helper.to_array(const_nd.attribute[0].t))
            if index != -1:
                name = index2name[index]
                onnx_config[nd.name] = config[name]
            nd.input[0] = mul_nd.input[0]
            # Remove constant node and multiply node
            model_onnx.graph.node.remove(const_nd)
            model_onnx.graph.node.remove(mul_nd)
            idx = idx-2
        idx = idx+1
    return model_onnx, onnx_config

def get_calib(model_onnx, onnx_config, calib_path):
    """
    Get calib from onnx_config, and convert float to hex.

    Parameters
    ----------
    model_onnx : onnx model
        The model_onnx prepared to convert
    onnx_config : dict
        The config which store the information of quantization. Like Op_name: tracked_min_ouput
    calib_path : str
        The path of the calib

    Returns
    -------
    model_calib : dict
        The dict which store the information of quantization. Like: Tensor_name: scale(float)
    """
    model_calib = {}
    onnx_nodes = model_onnx.graph.node
    
    for node in onnx_nodes:
        if node.name in onnx_config:
            if "tracked_min_input" in onnx_config[node.name] and "tracked_max_input" in onnx_config[node.name]:
                tracked_min_input = onnx_config[node.name]['tracked_min_input']
                tracked_max_input = onnx_config[node.name]['tracked_max_input']
                model_calib[node.input[0]] = (tracked_max_input - tracked_min_input) / 255
            
            if "tracked_min_output" in onnx_config[node.name] and "tracked_max_output" in onnx_config[node.name]:
                tracked_min_output = onnx_config[node.name]['tracked_min_output']
                tracked_max_output = onnx_config[node.name]['tracked_max_output']
                model_calib[node.output[0]] = (tracked_max_output - tracked_min_output) / 255
    
    calibrator_cache = open(calib_path, "w")
    calibrator_cache.write("TRT-7134-EntropyCalibration2\n")

    for name in model_calib.keys():
        calibrator_cache.write("{:s}: {}\n".format(
            name, hex(struct.unpack('<I', struct.pack('<f', model_calib[name]))[0])[2:]))

    return model_calib


def get_onnx_calib(model, config, input_shape, onnx_path, calib_path, input_names, output_names):
    """
    Convert torch model to onnx model and get TensorRT calib.

    Parameters
    ----------
    model : pytorch model
        The model to speed up by quantization
    config : dict
        Config recording bits number and name of layers
    input_shape : tuple
        The input shape of model, shall pass it to torch.onnx.export
    model_path : str
        The path user want to store onnx model which is converted from pytorch model
    calib_path : str
        The path user want to store model calib
    input_names : list
        Input name of onnx model providing for torch.onnx.export to generate onnx model
    output_name : list
        Output name of onnx model providing for torch.onnx.export to generate onnx model

    Returns
    -------
    onnx model
        Onnx model which is converted from pytorch model
    dict
        The configuration of onnx model layers and calibration parameters
    """
    # Support Gemm, Conv, Relu, Clip(Relu6) and MaxPool
    support_op = [torch.nn.Conv2d, torch.nn.Linear, torch.nn.ReLU, torch.nn.ReLU6, torch.nn.MaxPool2d]
    # Transfer bits number to onnx layer by using wrapper
    index2name = {}
    name2index = {}
    if config is not None:
        for i, name in enumerate(config.keys()):
            index2name[i] = name
            name2index[name] = i
    for name, module in model.named_modules():
        if config is not None and name in config:
            assert type(module) in support_op
            wrapper_module = LayernameModuleWrapper(module, name2index[name])
            _setattr(model, name, wrapper_module)
        elif type(module) in support_op:
            wrapper_module = LayernameModuleWrapper(module, -1)
            _setattr(model, name, wrapper_module)
    # Convert torch model to onnx model and save it in model_path
    dummy_input = torch.randn(input_shape)
    model.to('cpu')
    torch.onnx.export(model, dummy_input, onnx_path, verbose=False, input_names=input_names, output_names=output_names, export_params=True)

    # Load onnx model
    model_onnx = onnx.load(onnx_path)
    model_onnx, onnx_config = unwrapper(model_onnx, index2name, config) # ???torch???config???onnx???config
    onnx.save(model_onnx, onnx_path)

    onnx.checker.check_model(model_onnx)

    model_calib = get_calib(model_onnx, onnx_config, calib_path)

    return model_onnx, onnx_config

def torch_to_onnx(model, config, input_shape, model_path, input_names, output_names):
    """
    Convert torch model to onnx model and get layer bits config of onnx model.

    Parameters
    ----------
    model : pytorch model
        The model to speed up by quantization
    config : dict
        Config recording bits number and name of layers
    input_shape : tuple
        The input shape of model, shall pass it to torch.onnx.export
    model_path : str
        The path user want to store onnx model which is converted from pytorch model
    input_names : list
        Input name of onnx model providing for torch.onnx.export to generate onnx model
    output_name : list
        Output name of onnx model providing for torch.onnx.export to generate onnx model

    Returns
    -------
    onnx model
        Onnx model which is converted from pytorch model
    dict
        The configuration of onnx model layers and calibration parameters
    """
    # Support Gemm, Conv, Relu, Clip(Relu6) and MaxPool
    support_op = [torch.nn.Conv2d, torch.nn.Linear, torch.nn.ReLU, torch.nn.ReLU6, torch.nn.MaxPool2d]
    # Transfer bits number to onnx layer by using wrapper
    index2name = {}
    name2index = {}
    if config is not None:
        for i, name in enumerate(config.keys()):
            index2name[i] = name
            name2index[name] = i
    for name, module in model.named_modules():
        if config is not None and name in config:
            assert type(module) in support_op
            wrapper_module = LayernameModuleWrapper(module, name2index[name])
            _setattr(model, name, wrapper_module)
        elif type(module) in support_op:
            wrapper_module = LayernameModuleWrapper(module, -1)
            _setattr(model, name, wrapper_module)
    # Convert torch model to onnx model and save it in model_path
    dummy_input = torch.randn(input_shape)
    model.to('cpu')
    torch.onnx.export(model, dummy_input, model_path, verbose=False, input_names=input_names, output_names=output_names, export_params=True)

    # Load onnx model
    model_onnx = onnx.load(model_path)
    model_onnx, onnx_config = unwrapper(model_onnx, index2name, config) # ???torch???config???onnx???config
    onnx.save(model_onnx, model_path)

    onnx.checker.check_model(model_onnx)
    return model_onnx, onnx_config