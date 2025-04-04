set -e
DATASET_DIR=/media/scannetpp/0a5c013435/dslr/
PREPROCESSED_FN=image_undistorted_fisheye_fov7
STEP=5e-3
FOVMOD=1.1

python prepare_fov.py --path "$DATASET_DIR" --dst "$PREPROCESSED_FN" --step $STEP --fov_mod $FOVMOD

python train.py -s "$DATASET_DIR" -m output/scannetpp/0a5c013435 \
    --iterations 15000 \
    --checkpoint_iterations 200 1000 2000 3000 4000 7000 8000 9000 10000 12000 15000\
    --save_iterations 200 1000 2000 3000 4000 7000 8000 9000 10000 12000 15000\
    --test_iterations 200 1000 2000 3000 4000 7000 8000 9000 10000 12000 15000\
    --resolution 1 \
    --eval \
    --mask_path "$DATASET_DIR$PREPROCESSED_FN" \
    --sample_step $STEP --fov_mod $FOVMOD