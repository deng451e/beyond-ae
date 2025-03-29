

export CUDA_LAUNCH_BLOCKING=0 # Debug use 
export CUDA_VISIBLE_DEVICES=0 # set device to run 
 
echo "Select an model to run:"
echo "1.vicuna-7b-v1.5-16k"
echo "2.vicuna-13b-v1.3"
echo "3.lotus-12B"
echo "4.opt-1.3b"
echo "5.opt-6.7b"
echo "6.opt-13b"
read -p "Enter your choice [1-6]: " choice1

 
case $choice1 in
    1) model="lmsys/vicuna-7b-v1.5-16k";;
    2) model="lmsys/vicuna-13b-v1.3";;
    3) model="hakurei/lotus-12B";;
    4) model="facebook/opt-1.3b";;
    5) model="facebook/opt-6.7b";;
    6) model="facebook/opt-13b";;
    *) echo "Invalid choice";;
esac
echo "============================="
echo "Select an dataset to test:"
echo "1.open-instruct-v1"
echo "2.mt_bench.jsonl"
echo "3.content_rephrasing"
read -p "Enter your choice [1-3]: " choice2
case $choice2 in
    1) dataset="hakurei/open-instruct-v1";;
    2) dataset="data/mt_bench.jsonl";;
    3) dataset="facebook/content_rephrasing";;
    *) echo "Invalid choice";;
esac
echo "============================="
IFS='/' read -r -a hold <<< "$model"
config_file="${hold[1]}"   

python run_model.py --model_name_or_path $model  --data_root $dataset  --config_file_path "config/$config_file.json"   