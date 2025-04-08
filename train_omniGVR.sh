set -e
SCENE_ID=0a5c013435
DATASET_DIR=/media/scannetpp/demo/$SCENE_ID/dslr/
PREPROCESSED_DIR=undistorted_fovmaps/
REMAPPED_DIR=remapped_fisheye/
STEP=2e-3
FOVMOD=1.3
MASK_FN=fov_"$FOVMOD"_step_"$STEP"_mask.png
CAM_FN=colmap/cameras_fish.txt

python prepare_fov.py --path $DATASET_DIR --dst $PREPROCESSED_DIR --step $STEP --fov_mod $FOVMOD --mask_dst $MASK_FN

python train.py -s "$DATASET_DIR" -m output/scannetpp/$SCENE_ID \
    --iterations 30000 \
    --checkpoint_iterations 200 300 500 700 1000 2000 3000 4000 7000 8000 9000 10000 12000 15000 17000 20000 22000 25000 27000 30000\
    --save_iterations 200 300 500 700 1000 2000 3000 4000 7000 8000 9000 10000 12000 15000 17000 20000 22000 25000 27000 30000\
    --test_iterations 200 300 500 700 1000 2000 3000 4000 7000 8000 9000 10000 12000 15000 17000 20000 22000 25000 27000 30000\
    --resolution 1 \
    --eval \
    --mask_path $DATASET_DIR$PREPROCESSED_DIR$MASK_FN \
    --sibr_mask_refcam $DATASET_DIR$CAM_FN \
    --sample_step $STEP --fov_mod $FOVMOD

python extract_kb.py --path $DATASET_DIR --src $PREPROCESSED_DIR --dst $REMAPPED_DIR --step $STEP --fov_mod $FOVMOD