#!/usr/bin/python3

# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import datetime
import logging as log
import sys
from collections import defaultdict
from typing import Union
import numpy as np
from datetime import datetime

import openvino as ov
import openvino.properties.hint as hints

try:
    from openvino.runtime import Core, Model, CompiledModel, InferRequest, Output, get_version
except Exception as e:
    exception_type = type(e).__name__
    print(f"The following error happened while importing OpenVINO Python API module:\n[ {exception_type} ] {e}")
    sys.exit(1)

from openvino.tools.cross_check_tool.utils import get_config_dictionary, get_ops_list, print_output_ops, \
    input_processing, accuracy_metrics, validate_args, build_parser, set_logger, find_out_cct_mode, \
    print_all_over_the_net_metrics, update_global_accuracy_matrics, tensor_counters, performance_metrics, \
    dump_output_file, load_dump, error_handling, print_inputs, set_verbosity, perf_counts_to_dump, load_profiling_info


###
#   PLUGIN
###


@error_handling('plugin of \'{device}\' device config \'{config}\' loading')
def set_plugin_config(core: Core, device: str, config: str = None):
    core.set_property(device, get_config_dictionary(config_file=config))


@error_handling('\'{cpu_ext}\' cpu extensions loading')
def set_cpu_extensions(core: Core, cpu_ext: str):
    core.add_extension(cpu_ext)


def get_plugin(device: str, cpu_ext: str = None, config: str = None):
    core = Core()
    # log.info('{} plugin:\n          API version ............ {}'.format(device, plugin.version), extra={'no_lvl': True})
    set_plugin_config(core=core, device=device, config=config)
    if cpu_ext and 'CPU' in device:
        set_cpu_extensions(core=core, cpu_ext=cpu_ext)
    return core


###
#   MODEL
###


@error_handling('reading {xml_path} IR model')
def get_model(model_path: str, core: Core):
    model = core.read_model(model=model_path)
    # TODO: can we support it?
    if model.is_dynamic():
        print("Cross check tool doesn't support dynamic models for now.")
        # raise Exception("Cross check tool doesn't support dynamic models for now.")
    return model


@error_handling('compiling model for {device} device')
def get_compiled_model(core: Core, model: Model, device: str):
    return core.compile_model(model=model, device_name=device)


@error_handling('creating infer request')
def get_infer_request(compiled_model: CompiledModel):
    return compiled_model.create_infer_request()


@error_handling('output \'{output}\' addition for network from model \'{model}\'')
def get_model_copy_with_output(model: str, output: tuple, core: Core):
    model_copy = get_model(model_path=model, core=core)
    new_output = None
    if output not in ['None', None]:
        new_output = model_copy.add_outputs(output).pop()
    return model_copy, new_output

@error_handling('output \'{output}\' addition for network from model \'{model}\'')
def get_model_copy_with_output_multibatch(model: str, output: tuple, core: Core, inputs):
    model_copy = get_model(model_path=model, core=core)
    reshape(model_copy, inputs)
    new_output = None
    if output not in ['None', None]:
        new_output = model_copy.add_outputs(output).pop()
    return model_copy, new_output

@error_handling('getting model operations info')
def get_model_info(model: Model):
    return model.get_ordered_ops(), model.inputs, model.outputs


def check_inputs_and_default_outputs_are_equal(model, ref_model):
    if len(model.inputs) != len(ref_model.inputs):
        raise Exception("Models have different number of inputs! Cannot cross check!")
    if len(model.outputs) != len(ref_model.outputs):
        raise Exception("Models have different number of outputs! Cannot cross check!")
    for input, ref_input in zip(model.inputs, ref_model.inputs):
        if input.any_name != ref_input.any_name:
            raise Exception("Models have different inputs! Cannot cross check!")
    for output, ref_output in zip(model.outputs, ref_model.outputs):
        if output.any_name != ref_output.any_name:
            raise Exception("Models have different outputs! Cannot cross check!")


def get_ops_intersection(ops, ref_ops):
    ops_map = {node.friendly_name: node for node in ops}
    operation_names = set(ops_map.keys())
    ref_operation_names = set(node.friendly_name for node in ref_ops)
    intersection_names = operation_names.intersection(ref_operation_names)
    return [ops_map[intersection_name] for intersection_name in intersection_names]


def get_ops_union(ops, ref_ops):
    ops_map = {}
    for op, ref_op in zip(ops, ref_ops):
        ops_map.update({op.friendly_name: op})
        ops_map.update({ref_op.friendly_name: ref_op})
    return ops_map.values()

###
#   INFER
###


@error_handling('getting inference results for output: \'{output.any_name}\'')
def get_infer_results(infer_request: InferRequest, output: Output):
    return infer_request.get_tensor(output).data


@error_handling('getting performance counts from infer request')
def get_profiling_info(infer_request: InferRequest, port: Output):
    for pi in infer_request.profiling_info:
       if pi.node_name == port.node.friendly_name:
           return pi


@error_handling('processing inference on \'{device}\' device')
def infer(model: Model, core: Core, device: str, inputs: Union[list, dict], output=None):
    compiled_model = get_compiled_model(core=core, model=model, device=device)
    infer_request = get_infer_request(compiled_model)
    infer_request.infer(inputs)
    if output:
        result = get_infer_results(infer_request, output)
        prof_info = get_profiling_info(infer_request, output)
        helper = dict()
        for i, j in enumerate(zip(infer_request.model_outputs, infer_request.output_tensors)):
            helper[i] = j
        return result, prof_info, helper


@error_handling('computing overall performance')
def overall_accuracy_check(model: str, ref_model: str, out_ops: list, ref_out_ops: list, inputs: list,
                           ref_inputs: list, core: Core, device: str, ref_core: Core, ref_device: str, layers: str,
                           num_of_iterations: int):
    global_times, ref_global_times = [], []
    if layers in ['None', None]:
        model_copy, _ = get_model_copy_with_output(model=model, output=layers, core=core)
        ref_model_copy, _ = get_model_copy_with_output(model=ref_model, output=layers, core=ref_core)
        for i in range(num_of_iterations):
            t1 = datetime.datetime.now()
            infer(model=model_copy, core=core, device=device, inputs=inputs)
            t2 = datetime.datetime.now()
            infer(model=ref_model_copy, core=ref_core, device=ref_device, inputs=ref_inputs)
            t3 = datetime.datetime.now()
            global_times.append(t2 - t1)
            ref_global_times.append(t3 - t2)
    return global_times, ref_global_times


def reshape(network, input):
    batch_dim = 0
    if isinstance(input, list):
        batch = input[0].shape[0]
    else:
        batch = input[network.inputs[0].names.pop()].shape[batch_dim]
    input_shapes = {}
    for network_input in network.inputs:
        input_name = network_input.get_any_name()
        input_shapes[input_name] = network_input.get_partial_shape()
        input_shapes[input_name][batch_dim] = batch
        # for dim in range(network_input.get_partial_shape().rank.max_length):
        #     input_shapes[input_name][dim] = input[input_name].shape[dim]
    network.reshape(input_shapes)

def print_value(out_array, ref_array):
    import torch
    out_tensor = torch.tensor(out_array)
    ref_tensor = torch.tensor(ref_array)
    out_tensor_sq = out_tensor.flatten()
    ref_tensor_sq = ref_tensor.flatten()
    print(f"device tensor value : {out_tensor_sq[:10]}")
    print(f"refere tensor value : {ref_tensor_sq[:10]}")


def one_ir_mode(args):
    core = get_plugin(args.device, args.l, args.config)
    core.set_property("GPU", {hints.inference_precision: ov.runtime.Type.f32})
    model = get_model(model_path=args.model, core=core)
    model_ops, model_inputs, model_outputs = get_model_info(model)
    log.info(f'{args.device} vs {args.reference_device}')
    log.info(f'The same IR on both devices: {args.model}')
    out_ops = get_ops_list(model_ops, model_outputs, args.layers, args.op_list)
    print_inputs(model_inputs)
    print_output_ops(out_ops)
    ref_core = get_plugin(args.reference_device, args.l, args.reference_config)
    global_accuracy = []
    inputs = input_processing(model_path=args.model, model_inputs=model_inputs, input_file=args.input, batch_size=args.batch)
    reshape(model, inputs)
    global_times, ref_global_times = overall_accuracy_check(model=args.model, ref_model=args.model,
                                                            out_ops=out_ops, ref_out_ops=out_ops,
                                                            inputs=inputs, ref_inputs=inputs, core=core,
                                                            device=args.device, ref_core=ref_core,
                                                            ref_device=args.reference_device, layers=args.layers,
                                                            num_of_iterations=args.num_of_iterations)
    cnt = 0
    for op in reversed(out_ops):
        log.info(f'Layer {op.friendly_name} at cnt={cnt} statistics')
        # if(op.friendly_name!="Loop_210668"):
        if(cnt < 1):
            cnt+=1
            continue
        cnt+=1
        for i in range(op.get_output_size()):
            if op.get_output_size() > 1:
                log.info(f'Port {i}: ')
            try:
                model_copy, new_output = get_model_copy_with_output(model=args.model, output=(op.friendly_name, i), core=core)
                reshape(model_copy, inputs)
                out_tensor, pc, helper = infer(model=model_copy, core=core, device=args.device, inputs=inputs, output=new_output)
                ref_out_tensor, ref_pc, _ = infer(model=model_copy, core=ref_core, device=args.reference_device, inputs=inputs, output=new_output)
                a_m = accuracy_metrics(out_tensor, ref_out_tensor)
                performance_metrics(args.device, pc, args.reference_device, ref_pc)
                tensor_counters(out_tensor, ref_out_tensor)
                global_accuracy = update_global_accuracy_matrics(global_accuracy=global_accuracy, current_accuracy=a_m)
                print_value(out_tensor, ref_out_tensor)
                return
            except Exception as e:
                log.error(f"layer No. {cnt} = {op.friendly_name} throw an except {e}")
    print_all_over_the_net_metrics(global_times=global_times, ref_global_times=ref_global_times,
                                   global_accuracy=global_accuracy)

def one_ir_two_batch_mode(args):
    core = get_plugin(args.device, args.l, args.config)
    core.set_property("GPU", {hints.inference_precision: ov.runtime.Type.f32})
    model = get_model(model_path=args.model, core=core)
    model_ops, model_inputs, model_outputs = get_model_info(model)
    inputs = input_processing(model_path=args.model, model_inputs=model_inputs, input_file=args.input, batch_size=args.batch)
    reshape(model, inputs)
    log.info(f'{args.device} vs {args.reference_device}')
    log.info(f'The same IR on both devices: {args.model}')
    out_ops = get_ops_list(model_ops, model_outputs, args.layers, args.op_list)
    print_inputs(model_inputs)
    print_output_ops(out_ops)
    ref_core = get_plugin(args.reference_device, args.l, args.reference_config)
    global_accuracy = []
    global_times, ref_global_times = overall_accuracy_check(model=args.model, ref_model=args.model,
                                                            out_ops=out_ops, ref_out_ops=out_ops,
                                                            inputs=inputs, ref_inputs=inputs, core=core,
                                                            device=args.device, ref_core=ref_core,
                                                            ref_device=args.reference_device, layers=args.layers,
                                                            num_of_iterations=args.num_of_iterations)
    cnt = 0
    for op in reversed(out_ops):
        log.info(f'Layer {cnt} =  {op.friendly_name} statistics')
        # if(op.friendly_name!="Transpose_2915385"):
        # if(cnt < 105):
        #     cnt+=1
        #     continue
        cnt+=1
        for i in range(op.get_output_size()):
            if op.get_output_size() > 1:
                log.info(f'Port {i}: ')
            try:
                model_copy, new_output = get_model_copy_with_output_multibatch(model=args.model, output=(op.friendly_name, i), core=core, inputs=inputs)
                if(new_output.partial_shape.is_static):
                    if(new_output.shape.to_string() == "[]" or new_output.shape[0] != 2):
                        log.info(f'Layer {op.friendly_name} is not applicable by shape {new_output.shape}.')
                        continue
                # reshape(model_copy, inputs)
                out_tensor, pc, helper = infer(model=model_copy, core=core, device=args.device, inputs=inputs, output=new_output)
                # ref_out_tensor, ref_pc = infer(model=model_copy, core=ref_core, device=args.reference_device, inputs=inputs, output=new_output)
                log.info(f'Layer {op.friendly_name} is now running with shape {out_tensor.shape}.')
                
                if(out_tensor.shape[0] != 2):
                    log.info(f'Layer {op.friendly_name} is not applicable by dynamic shape {out_tensor.shape}.')
                    continue
                if(len(out_tensor) == 2):
                    ref_out_tensor = out_tensor[1]
                    out_tensor = out_tensor[0]
                else:
                    ref_out_tensor = out_tensor[1,:]
                    out_tensor = out_tensor[0,:]
                ref_pc = pc
                a_m = accuracy_metrics(out_tensor, ref_out_tensor)
                performance_metrics(args.device, pc, args.reference_device, ref_pc)
                tensor_counters(out_tensor, ref_out_tensor)
                print_value(out_tensor, ref_out_tensor)
                global_accuracy = update_global_accuracy_matrics(global_accuracy=global_accuracy, current_accuracy=a_m)
                return
            except Exception as e:
                log.error(f"layer No. {cnt} = {op.friendly_name} throw an except {e}")
    print_all_over_the_net_metrics(global_times=global_times, ref_global_times=ref_global_times,
                                   global_accuracy=global_accuracy)

def one_ir_two_batch_compare_one_mode(args):
    core = get_plugin(args.device, args.l, args.config)
    model = get_model(model_path=args.model, core=core)
    model_ops, model_inputs, model_outputs = get_model_info(model)
    inputs_b2 = input_processing(model_path=args.model, model_inputs=model_inputs, input_file=args.input, batch_size=args.batch)
    inputs_b1 = input_processing(model_path=args.model, model_inputs=model_inputs, input_file=args.input, batch_size=1)
    reshape(model, inputs_b2)
    log.info(f'{args.device} vs {args.reference_device}')
    log.info(f'The same IR on both devices: {args.model}')
    out_ops = get_ops_list(model_ops, model_outputs, args.layers)
    print_inputs(model_inputs)
    print_output_ops(out_ops)
    ref_core = get_plugin(args.reference_device, args.l, args.reference_config)
    global_accuracy = []
    global_times, ref_global_times = overall_accuracy_check(model=args.model, ref_model=args.model,
                                                            out_ops=out_ops, ref_out_ops=out_ops,
                                                            inputs=inputs_b2, ref_inputs=inputs_b1, core=core,
                                                            device=args.device, ref_core=ref_core,
                                                            ref_device=args.reference_device, layers=args.layers,
                                                            num_of_iterations=args.num_of_iterations)
    cnt = 0
    for op in out_ops:
        log.info(f'Layer {cnt} =  {op.friendly_name} statistics')
        if(op.friendly_name!="Loop_210668"):
        # if(cnt < 3020):
            cnt+=1
            continue
        cnt+=1
        for i in range(op.get_output_size()):
            if op.get_output_size() > 1:
                log.info(f'Port {i}: ')
            model_copy_b2, new_output_b2 = get_model_copy_with_output_multibatch(model=args.model, output=(op.friendly_name, i), core=core, inputs=inputs_b2)
            # reshape(model_copy_b2, inputs_b2)
            model_copy_b1, new_output_b1 = get_model_copy_with_output_multibatch(model=args.model, output=(op.friendly_name, i), core=core, inputs=inputs_b1)
            # reshape(model_copy_b1, inputs_b1)
            if(new_output_b2.partial_shape.is_static):
                if(new_output_b2.shape.to_string() == "[]" or new_output_b2.shape[0] != 2):
                    log.info(f'Layer {op.friendly_name} is not applicable by shape {new_output_b2.shape}.')
                    continue
            # reshape(model_copy, inputs)
            out_tensor, pc = infer(model=model_copy_b2, core=core, device=args.device, inputs=inputs_b2, output=new_output_b2)
            out_tensor_ref, pc_ref = infer(model=model_copy_b1, core=core, device=args.reference_device, inputs=inputs_b1, output=new_output_b1)
            # ref_out_tensor, ref_pc = infer(model=model_copy, core=ref_core, device=args.reference_device, inputs=inputs, output=new_output)
            log.info(f'Layer {op.friendly_name} is now running with shape {out_tensor.shape}.')
            
            if(out_tensor.shape[0] != 2):
                log.info(f'Layer {op.friendly_name} is not applicable by dynamic shape {out_tensor.shape}.')
                continue
            if(len(out_tensor) == 2):
                b2_out_tensor = out_tensor
                out_tensor = out_tensor
                log.info(f'Layer {op.friendly_name} may not applicable by dynamic shape {out_tensor.shape} in batch 2 and {out_tensor_ref.shape} in batch 1.')
                # continue
            else:
                b2_out_tensor = out_tensor[1,:]
                out_tensor = out_tensor[0,:]
            a_m = accuracy_metrics(b2_out_tensor, out_tensor_ref)
            performance_metrics(args.device, pc, args.reference_device, pc_ref)
            tensor_counters(b2_out_tensor, out_tensor_ref)
            global_accuracy = update_global_accuracy_matrics(global_accuracy=global_accuracy, current_accuracy=a_m)
    print_all_over_the_net_metrics(global_times=global_times, ref_global_times=ref_global_times,
                                   global_accuracy=global_accuracy)

def run_once(args):
    print("step into run once mode.")
    core = get_plugin(args.device, args.l, args.config)
    model = get_model(model_path=args.model, core=core)
    model_ops, model_inputs, model_outputs = get_model_info(model)
    inputs = input_processing(model_path=args.model, model_inputs=model_inputs, input_file=args.input, batch_size=args.batch)
    reshape(model, inputs)
    log.info(f'{args.device} vs {args.reference_device}')
    log.info(f'The same IR on both devices: {args.model}')
    out_ops = get_ops_list(model_ops, model_outputs, args.layers)
    print_inputs(model_inputs)
    print_output_ops(out_ops)
    
    print("inference starting.")

    out_tensor, pc = infer(model=model, core=core, device=args.device, inputs=inputs)
    print("execute finish.")

def two_ir_mode(args):
    core = get_plugin(args.device, args.l, args.config)
    ref_core = get_plugin(args.reference_device, args.l, args.reference_config)
    model = get_model(model_path=args.model, core=core)
    model_ops, model_inputs, model_outputs = get_model_info(model)
    ref_model = get_model(model_path=args.reference_model, core=ref_core)
    ref_model_ops, _, _ = get_model_info(ref_model)
    check_inputs_and_default_outputs_are_equal(model, ref_model)
    log.info(f'{args.device} vs {args.reference_device}')
    log.info(f'IR for {args.device} : {args.model}')
    log.info(f'IR for {args.reference_device} : {args.reference_model}')
    if args.reference_layers:
        out_ops = get_ops_list(model_ops, model_outputs, args.layers)
        ref_out_ops = get_ops_list(ref_model_ops, model_outputs, args.reference_layers)
        if len(out_ops) != len(ref_out_ops):
            raise Exception("Number of layers to compare against should be equal!")
    else:
        ref_out_ops = out_ops = get_ops_list(get_ops_intersection(model_ops, ref_model_ops), model_outputs, args.layers)
    print_inputs(model_inputs)
    print_output_ops(get_ops_union(out_ops, ref_out_ops))
    inputs = input_processing(model_path=args.model, model_inputs=model_inputs, input_file=args.input)
    global_accuracy = []
    global_times, ref_global_times = overall_accuracy_check(model=args.model, ref_model=args.reference_model,
                                                            out_ops=out_ops, ref_out_ops=out_ops,
                                                            inputs=inputs_b2, ref_inputs=inputs_b1, core=core,
                                                            device=args.device, ref_core=ref_core,
                                                            ref_device=args.device, layers=args.layers,
                                                            num_of_iterations=args.num_of_iterations)
    for op, ref_op in zip(out_ops, ref_out_ops):
        if op.friendly_name == ref_op.friendly_name:
            log.info(f'Layer {op.friendly_name} statistics')
        else:
            if op.get_output_size() != ref_op.get_output_size():
                log.warning(f"Skipping {op.friendly_name} vs {ref_op.frinedly_name} comparison due to different number of outputs!")
                continue
            log.info(f'Layer {op.friendly_name} vs {ref_op.friendly_name} statistics')
        for i in range(op.get_output_size()):
            if op.get_output_size() > 1:
                log.info(f'Port {i}: ')
            model_copy, new_output = get_model_copy_with_output(model=args.model, output=(op.friendly_name, i), core=core)
            ref_model_copy, ref_new_output = get_model_copy_with_output(model=args.reference_model, output=(ref_op.friendly_name, i), core=ref_core)
            out_tensor, pc = infer(model=model_copy, core=core, device=args.device, inputs=inputs, output=new_output)
            ref_out_tensor, ref_pc = infer(model=ref_model_copy, core=ref_core, device=args.reference_device,
                                                                    inputs=inputs, output=ref_new_output)
            a_m = accuracy_metrics(out_tensor, ref_out_tensor)
            performance_metrics(args.device, pc, args.reference_device, ref_pc)
            tensor_counters(out_tensor, ref_out_tensor)
            global_accuracy = update_global_accuracy_matrics(global_accuracy=global_accuracy, current_accuracy=a_m)
    print_all_over_the_net_metrics(global_times=global_times, ref_global_times=ref_global_times,
                                   global_accuracy=global_accuracy)


def dump_mode(args):
    core = get_plugin(args.device, args.l, args.config)
    model = get_model(model_path=args.model, core=core)
    model_ops, model_inputs, model_outputs = get_model_info(model)
    out_ops = get_ops_list(model_ops, model_outputs, args.layers)
    inputs = input_processing(args.model, model_inputs, args.input)
    dump_dict = defaultdict(list)
    for op in out_ops:
        for i in range(op.get_output_size()):
            if op.get_output_size() > 1:
                log.info(f'Layer {op.friendly_name}, port {i} processing')
            else:
                log.info(f'Layer {op.friendly_name} processing')
            model_copy, new_output = get_model_copy_with_output(model=args.model, output=(op.friendly_name, i), core=core)
            out_tensor, pc = infer(model=model_copy, core=core, device=args.device, inputs=inputs, output=new_output)
            dump_dict[op.friendly_name].append(np.array({'tensor': out_tensor, 'pc': perf_counts_to_dump(pc)}))
    dump_dict["device"] = args.device
    dump_output_file(args.model + '_' + args.device + '_dump.npz', dump_dict)


def load_mode(args):
    core = get_plugin(args.device, args.l, args.config)
    log.info(f'IR for {args.device} : {args.model}')
    log.info(f'Loading tensors from {args.load}')
    model = get_model(model_path=args.model, core=core)
    model_ops, model_inputs, model_outputs = get_model_info(model)
    out_ops = get_ops_list(model_ops, model_outputs, args.layers)
    print_inputs(model_inputs)
    print_output_ops(out_ops)
    inputs = input_processing(args.model, model_inputs, args.input)
    global_accuracy = []
    loaded = load_dump(args.load)
    for op in out_ops:
        if op.friendly_name in loaded:
            log.info(f'Layer {op.friendly_name} statistics')
        else:
            log.info(f'Statistics for layer \'{op.friendly_name}\' was not dumped. Skipping this layer.')
            continue
        for i in range(op.get_output_size()):
            if op.get_output_size() > 1:
                log.info(f'Port {i}: ')
            model_copy, new_output = get_model_copy_with_output(model=args.model, output=(op.friendly_name, i), core=core)
            out_tensor, pc = infer(model=model_copy, core=core, device=args.device, inputs=inputs, output=new_output)
            ref_out_tensor, ref_pc = loaded[op.friendly_name][i]['tensor'], load_profiling_info(loaded[op.friendly_name][i]['pc'])
            a_m = accuracy_metrics(out_tensor, ref_out_tensor)
            performance_metrics(args.device, pc, loaded["device"], ref_pc)
            tensor_counters(out_tensor, ref_out_tensor)
            global_accuracy = update_global_accuracy_matrics(global_accuracy=global_accuracy, current_accuracy=a_m)
    print_all_over_the_net_metrics(global_accuracy=global_accuracy)


def main():
    # Get current date and time
    now = datetime.now()

    # Format as a string
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    filename='./logfile_{}.log'.format(now_str)
    set_logger(log.DEBUG, filename)
    args = validate_args(build_parser().parse_args())

    log.info(f'OpenVINO:\n          API version ............ {get_version()}', extra={'no_lvl': True})
    set_verbosity(args.verbosity)
    mode = find_out_cct_mode(args)
    if mode == 1:
        log.info('Cross check with one IR was enabled')
        one_ir_mode(args)
    elif mode == 2:
        log.info('Cross check with two IRs was enabled')
        two_ir_mode(args)
    elif mode == 3:
        log.info('Dump mode was enabled')
        dump_mode(args)
    elif mode == 4:
        log.info('Load mode was enabled')
        load_mode(args)
    elif mode == 5:
        log.info('batch check with one IR was enabled')
        one_ir_two_batch_mode(args)
    elif mode == 6:
        log.info('batch check with 1 batch with one IR was enabled')
        one_ir_two_batch_compare_one_mode(args)
    elif mode == 7:
        log.info('No check with dry run was enabled')
        run_once(args)
    log.info("Execution successful")


if __name__ == '__main__':
    main()