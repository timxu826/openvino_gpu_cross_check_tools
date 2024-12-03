export OV_GPU_Verbose=1
export OV_GPU_DumpLayersPath=/media/NewDisk/jia3xu/openvino/CVS-143684/dump/single_layer/
export OV_GPU_DumpLayers="NonMaxSuppression_207083"
python cross_check_tool.py -i /media/NewDisk/jia3xu/openvino/CVS-143684/openvino/tools/cross_check_tool/my_arrays.npz -m /media/NewDisk/jia3xu/openvino/CVS-143684/dump/model/TF_Faster_RCNN_Inception_ResNet_v2_atrous_coco_api_2_True_batch_2_device_GPU_precision_FP32hp0g457d/model.xml -d GPU -ref_d CPU --layers all -b 2 -bc 1 -run
