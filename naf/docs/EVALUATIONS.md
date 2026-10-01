# 🔄 Evaluation 

You should put the loaded weight file `naf_release.pth` in the `output/` folder.

To evaluate the upsampler on semantic segmentation, simply run:
``` python
python evaluation/eval_seg_probing.py dataset=voc eval.model_ckpt=output/naf_release.pth
```
