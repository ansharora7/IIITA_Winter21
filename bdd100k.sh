CUDA_VISIBLE_DEVICES=3 python train.py --img 720 --cfg yolov5s.yaml --hyp hyp.scratch.yaml --batch 8 --epochs 20 --data bdd100k.yaml --weights yolov5s.pt --workers 24 --name yolo_road_det
