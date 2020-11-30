export PYTHONPATH="${PYTHONPATH}:/shared/xudongliu/code/f_style/lib"
export CUDA_VISIBLE_DEVICES=4,5,6,7
python segment.py train -d /shared/xudongliu/bdd100k/seg/seg -c 19 -s 768 --arch dla34up \
    --batch-size 8 --lr 0.01 --momentum 0.9 --lr-mode poly \
    --epochs 500 --bn-sync --random-scale 2 --random-rotate 0 \
    --random-color --pretrained-base imagenet -o out/segnet_new_500e