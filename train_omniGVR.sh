set -e
DATASET_DIR=/media/scannetpp/0a5c013435/dslr/
PREPROCESSED_DIR=image_undistorted_fisheye_fov7/
STEP=5e-3
FOVMOD=1.1
MASK_FN=fov_"$FOVMOD"_step_"$STEP"_mask.png

# cd ${baseline}/omni-gvr

python prepare_fov.py --path "$DATASET_DIR" --dst "$PREPROCESSED_DIR" --step $STEP --fov_mod $FOVMOD --mask_dst $MASK_FN

python train.py -s "$DATASET_DIR" -m output/scannetpp/0a5c013435 \
    --iterations 15000 \
    --checkpoint_iterations 200 1000 2000 3000 4000 7000 8000 9000 10000 12000 15000\
    --save_iterations 200 1000 2000 3000 4000 7000 8000 9000 10000 12000 15000\
    --test_iterations 200 1000 2000 3000 4000 7000 8000 9000 10000 12000 15000\
    --resolution 1 \
    --eval \
    --mask_path "$DATASET_DIR$PREPROCESSED_DIR$MASK_FN" \
    --sample_step $STEP --fov_mod $FOVMOD