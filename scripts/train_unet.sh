export PYTHONPATH="${PYTHONPATH}:/shared/xudongliu/code/f_style/lib"
export CUDA_VISIBLE_DEVICES=4,5,6,7
python segment_refinenet.py train -d /shared/xudongliu/bdd100k/seg/seg -c 19 -s 513 --arch dla34up \
    --batch-size 16 --lr 0.02 --momentum 0.9 --lr-mode poly \
    --epochs 500 --bn-sync --random-scale 2 --random-rotate 0 \
    --random-color --pretrained-base imagenet -o out/unet_16bs_0.02_513_500e