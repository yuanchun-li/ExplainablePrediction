LANG=python
DATADIR=/home/v-weixyan/Desktop/new/CodeXGLUE/Code-Code/CodeCompletion-line/dataset/py150/line_completion
LITFILE=/home/v-weixyan/Desktop/new/CodeXGLUE/Code-Code/CodeCompletion-line/dataset/py150/literals.json
OUTPUTDIR=/home/v-weixyan/Desktop/new/CodeXGLUE/Code-Code/CodeCompletion-line/save/pythonCorpus
LOGFILE=completion_pythonCorpus.log
PER_NODE_GPU=1

python -m torch.distributed.launch --nproc_per_node=$PER_NODE_GPU run_lm.py \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
	--log_file=$LOGFILE \
        --model_type=rnn \
        --block_size=1024 \
        --do_train \
        --gpu_per_node $PER_NODE_GPU \
        --learning_rate=8e-5 \
        --weight_decay=0.01 \
        --evaluate_during_training \
        --per_gpu_train_batch_size=2 \
        --per_gpu_eval_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --num_train_epochs=3 \
        --logging_steps=100 \
        --save_steps=1000 \
        --seed=42 \
        --overwrite_output_dir \
        --not_pretrain

