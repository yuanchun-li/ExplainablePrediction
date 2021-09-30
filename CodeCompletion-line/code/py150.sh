export CUDA_VISIBLE_DEVICES=0
LANG=python
DATADIR=/home/v-weixyan/Desktop/new/CodeXGLUE/Code-Code/CodeCompletion-line/dataset/py150/line_completion
LITFILE=/home/v-weixyan/Desktop/new/CodeXGLUE/Code-Code/CodeCompletion-line/dataset/py150/literals.json
OUTPUTDIR=/home/v-weixyan/Desktop/new/CodeXGLUE/Code-Code/CodeCompletion-line/save/pythonCorpus
PRETRAINDIR=/home/v-weixyan/Desktop/CodeGPT/checkpoint
LOGFILE=completion_pythonCorpus_eval.log

python -u run_lm.py \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=1024 \
        --eval_line \
        --logging_steps=100 \
        --seed=42
