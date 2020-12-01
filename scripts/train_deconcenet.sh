export PYTHONPATH="${PYTHONPATH}:/shared/xudongliu/code/f_style/lib"
export CUDA_VISIBLE_DEVICES=0,1,2,3
python segment_deconvnet.py train -d /shared/xudongliu/bdd100k/seg/seg -c 19 -s 769 --arch dla34up \
    --batch-size 8 --lr 0.01 --momentum 0.9 --lr-mode poly \
    --epochs 500 --bn-sync --random-scale 2 --random-rotate 0 \
    --random-color --pretrained-base imagenet -o out/convnet_224-769_500e 