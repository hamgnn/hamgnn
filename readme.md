
Oversmooth on the cora dataset:

```
python train_all.py --dataset cora --task nc --epochs 100 --patience 200 --dim 64 --num-layers 3 --cuda 0 --odemap h2extend --acth2 rehu --feat_h2_scale 16 --save 0;
python train_all.py --dataset cora --task nc --epochs 100 --patience 200 --dim 64 --num-layers 5 --cuda 0 --odemap h2extend --acth2 rehu --feat_h2_scale 16 --save 0;
python train_all.py --dataset cora --task nc --epochs 100 --patience 200 --dim 64 --num-layers 10 --cuda 0 --odemap h2extend --acth2 rehu --feat_h2_scale 16 --save 0;
python train_all.py --dataset cora --task nc --epochs 100 --patience 200 --dim 64 --num-layers 20 --cuda 0 --odemap h2extend --acth2 rehu --feat_h2_scale 16 --save 0;
python train_all.py --dataset cora --task nc --epochs 100 --patience 200 --dim 64 --num-layers 32 --cuda 0 --odemap h2extend --acth2 rehu --feat_h2_scale 16 --save 0;
python train_all.py --dataset cora --task nc --epochs 100 --patience 200 --dim 64 --num-layers 64 --cuda 0 --odemap h2extend --acth2 rehu --feat_h2_scale 16 --save 0;
```

