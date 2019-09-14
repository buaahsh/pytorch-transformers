# DIR=/mnt/localdata/users/shaohan/git/ecnet
DIR=/home/shaohan/git/ecnet
python examples/run_glue.py --model_type roberta \
 --model_name_or_path roberta-base \
 --task_name bow \
 --do_train \
 --data_dir $DIR \
 --max_seq_length 128 \
 --per_gpu_eval_batch_size 8 \
 --per_gpu_train_batch_size 64 \
 --learning_rate 2e-5 \
 --num_train_epochs 20.0 \
 --output_dir $DIR/model_v1 \
 --logging_steps 100 \
 --save_steps 10000 \
 --fp16
