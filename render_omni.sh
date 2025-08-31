set -e
DATASET_DIR=/home/choyingw/Documents/ODGS/datasets/Ricoh360/bricks/
#PREPROCESSED_FN=image_undistorted_fisheye_fov7
STEP=3e-3
FOVMOD=1.0
ITERATIONS=15000
OUTPUT_DIR=/home/choyingw/Documents/0403_clone/gaussian-splatting-archive/output/Ricoh360/bricks/
RAYMAP=/home/choyingw/Documents/ODGS/datasets/Ricoh360/bricks/raymap_fisheye.npy

python render.py -s "$DATASET_DIR" -m $OUTPUT_DIR \
    --iteration $ITERATIONS \
    --resolution 1 \
    --sample_step $STEP --fov_mod $FOVMOD \
    --skip_train \
    --mask_path "$DATASET_DIR"fovmaps_fov_"$FOVMOD"_step_"$STEP"/fov_"$FOVMOD"_step_"$STEP"_mask.png \
    --raymap_path $RAYMAP \
    --xi 1.0 \

#python3 extract_fov_360.py --path $OUTPUT_DIR/test/ours_$ITERATIONS --src renders --dst renders_remap --step $STEP