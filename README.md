# KTV
The code of [KTV](https://ojs.aaai.org/index.php/AAAI/article/view/37862): Keyframes and Key Tokens Selection for Efficient Training-Free Video LLMs, accepted by AAAI-2026

## conda env set 
```
conda create -n ktv python=3.10.12
conda activate ktv
cd ktv
cd llava
pip install -e .
pip install opencv-python numpy==1.26.2 protobuf 
pip install transformers_stream_generator
```
## Download pre-trained LLaVA-NeXT weights

```
git lfs clone https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b liuhaotian/llava-v1.6-vicuna-7b
git lfs clone https://huggingface.co/liuhaotian/llava-v1.6-34b liuhaotian/llava-v1.6-34b
```
Also you can download by yourself in
[liuhaotian/llava-v1.6-vicuna-7b](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b) and 
[liuhaotian/llava-v1.6-34b](https://huggingface.co/liuhaotian/llava-v1.6-34b)
## Inference
### 1. select keyframes
use 
```
keyframe_select_new.py
```
to extract Dinov2 features. Then use 
```
cluster_keyframe_and_order.py
```
to select keyframes for each test data.
Example:
```
python3 keyframe_select_new.py \
  --gt_file playground/gt_qa_files/STAR/val_qa.json \
  --video_dir your_video_dir \
  --output_pickle star_dino_features.pkl

python3 cluster_keyframe_and_order.py \
  --gt_file playground/gt_qa_files/STAR/val_qa.json \
  --feature_pickle star_dino_features.pkl \
  --output_json star_keyframes.json
```
### 2. Inference with llava
```
bash runcode.sh
e.g.
python3 run_inference_multiple_choice_qa.py --video_dir your_video_dir --gt_file your_dataset_gt_file --output_dir dataset --output_name your_output_name --model_path your_model_path --conv_mode multiple_choice_allvideo_34b_v4 --num_frames 6  --image_aspect_ratio resize --rope_scaling_factor 2 --key_frame_path nextqa_keyframe6_order.json --prune_mode cls_new_token_sim --rate 0.2 --tokens_num 1872 
--num_frames: the num of keyframes 
--rate: the alpha balancing the importance and redunancy for a vision token
--tokens_num: number of vision token sent to LLM
```
If you want  to use muliti-gpus you can add 
```
CUDA_VISIBLE_DEVICES=0,1,2,3 
```
before the python commmand "CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ..."
