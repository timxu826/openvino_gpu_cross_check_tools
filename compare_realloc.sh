#!/bin/bash

# Define the common variables
set -exo
Param_OV_GPU_Verbose="1"
Param_OV_GPU_DumpLayersPath="/media/NewDisk/jia3xu/openvino/CVS-156289/issued/dump/layers/apply_27780a6/"
Param_input_path="/media/NewDisk/jia3xu/openvino/CVS-156289/issued/data/inputs/data_2.pth"
Param_model_path="/media/NewDisk/jia3xu/openvino/CVS-156289/issued/model/Pytorch_Minicpm_2B_api_2_True_batch_2_device_GPU_precision_FP16_ylyolf1/model.xml"
Param_op_list_path="/media/NewDisk/jia3xu/openvino/CVS-156289/issued/dump/analysis_path/4227_mapping.pkl"

# Define the layers to dump
#？
# layers_to_dump=("fullyconnected:__module.model.layers.2.self_attn.k_proj/aten::linear/MatMul" "fullyconnected:__module.model.layers.2.self_attn.q_proj/aten::linear/MatMul" "fullyconnected:__module.model.layers.2.self_attn.v_proj/aten::linear/MatMul" "fullyconnected:__module.model.layers.2/aten::mul/Multiply_decompressed_to_f32" "fullyconnected:__module.model.layers.2.mlp.up_proj/aten::linear/MatMul" "fullyconnected:__module.model.layers.2.mlp.gate_proj/aten::linear/MatMul" "fullyconnected:__module.model.layers.2.mlp.gate_proj/aten::linear/MatMul" "fullyconnected:__module.model.layers.2/aten::mul/Multiply_1_decompressed_to_f32")
layers_to_dump=("__module.model/aten::masked_fill/Select" "__module.model/aten::masked_fill/Select_1" "__module.model/aten::to/Convert_3")

# Loop through each layer and execute the command
for layer in "${layers_to_dump[@]}"; do
    Param_OV_GPU_DumpLayers="$layer"
    log_file="log_$(echo $layer | tr -d ':/').txt" # Create a log filename from the layer name

    # Execute the command and save the output to the log file
    OV_GPU_Verbose=$Param_OV_GPU_Verbose OV_GPU_DumpLayers=$Param_OV_GPU_DumpLayers \
    OV_GPU_DumpLayersPath=$Param_OV_GPU_DumpLayersPath \
    python ./cross_check_tool.py \
        -i $Param_input_path \
        -m $Param_model_path \
        -d GPU -ref_d CPU --layers file --op_list $Param_op_list_path \
        2>&1 | tee $log_file
done
