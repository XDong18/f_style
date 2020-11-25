export PYTHONPATH="${PYTHONPATH}:/shared/xudongliu/code/f_style/lib"
export CUDA_VISIBLE_DEVICES=0,1,2,3
python segment.py train -e -d /shared/xudongliu/bdd100k/seg/seg -c 19 -s 768 --arch dla34up \
    --batch-size 8 --lr 0.01 --momentum 0.9 --lr-mode poly \
    --epochs 500 --bn-sync --random-scale 2 --random-rotate 0 \
    --random-color --pretrained-base imagenet -o out/segnet_500e --resume out/segnet_500e/checkpoint_73.pth.tar