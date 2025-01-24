# FLIR
python train.py --cfg ./models/transformer/yolov5l_Transfusion_FLIR_DeformDotAttnLocal.yaml --data ./data/multispectral/FLIR-align-3class.yaml --hyp ./data/hyp.scratch_FLIR.yaml --project saves/FLIR_DeformCrossAttn --name DeformCAT --epochs 15

# CVC-14
python train.py --cfg ./models/transformer/yolov5l_Transfusion_CVC_DeformDotAttnLocal.yaml --data ./data/multispectral/CVC-14.yaml --hyp ./data/hyp.scratch_CVC.yaml --project saves/CVC_DeformCrossAttn --name DeformCAT --epochs 30

# KAIST
python train.py --cfg ./models/transformer/yolov5l_Transfusion_kaist_DeformDotAttnLocal.yaml --data ./data/multispectral/KAIST.yaml --hyp ./data/hyp.scratch_KAIST.yaml --project saves/KAIST_DeformCrossAttn --name DeformCAT --epochs 60

# LLVIP
python train.py --cfg ./models/transformer/yolov5l_Transfusion_LLVIP_DeformDotAttnLocal.yaml --data ./data/multispectral/LLVIP.yaml --hyp ./data/hyp.scratch_LLVIP.yaml --project saves/LLVIP_DeformCrossAttn --name DeformCAT --epochs 30
