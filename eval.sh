set -e
SKIP_TRAIN=true

SCENE_ID=2a1a3afad9 #2a1a3afad9 1f7cbbdde1 4ef75031e3 1d003b07bd 0a5c013435
DATA_ROOT=/media/scannetpp/demo/
DATASET_DIR=$DATA_ROOT$SCENE_ID/dslr/
# OUTPUT_DIR=./output_ut_tight/scannetpp/$SCENE_ID
# OUTPUT_DIR=./output_ewa/scannetpp/$SCENE_ID
# OUTPUT_DIR=./output_fov/scannetpp_fov0.85/$SCENE_ID
# OUTPUT_DIR=./output_fov/scannetpp/$SCENE_ID
# OUTPUT_DIR=../../ablation/omni-gvr-scannetpp/output_fov_updated/scannetpp/$SCENE_ID
# OUTPUT_DIR=../../omni-3dgs/output_achive/scannetpp/$SCENE_ID
OUTPUT_DIR=../../Fisheye-GS-scalingup/output_scalingup/scannetpp/dslr/$SCENE_ID

STEP_TRAIN=0.002
STEP_EVAL=0.002

FOVMOD_TRAIN=1.3 #0.85 #0.85 #1.0 #1.3 #1.0 #1.3
FOVMOD_EVAL=2.0 #0.85 #0.85 #1.0 #2.0 #1.0 #2.0

DIST_SCALING=0.0
FOCAL_SCALING=0.6
MIRR_SHIFT=0.0
RENDER_MODEL=KB

FOVMAP_DIR_TRAIN=undistorted_fovmaps_fov_"$FOVMOD_TRAIN"_step_"$STEP_TRAIN"/
FOVMAP_DIR_EVAL=undistorted_fovmaps_fov_"$FOVMOD_EVAL"_step_"$STEP_EVAL"/

TRAIN_MASK_FN=fov_"$FOVMOD_TRAIN"_step_"$STEP_TRAIN"_mask.png
TEST_MASK_FN=fov_"$FOVMOD_EVAL"_step_"$STEP_EVAL"_mask.png

ITERS_NUM=20000

# # eval
# python prepare_fov.py --path $DATASET_DIR --dst $FOVMAP_DIR_EVAL --step $STEP_EVAL --fov_mod $FOVMOD_EVAL --mask_dst $TEST_MASK_FN
# python gridmap/scannetpp/create_kb_gridmap.py 

# python kb_raymap.py --path $DATASET_DIR \
#                     --step $STEP_EVAL --fov_mod $FOVMOD_EVAL --gridmap_restrict \
#                     # --distortion_scaling $DIST_SCALING \
#                     # --mirror_shift $MIRR_SHIFT \

# python gridmap/scannetpp/create_kb_raymap.py --path $DATA_ROOT \
#                                              --scenes $SCENE_ID \
#                                              --distortion_scaling $DIST_SCALING \
#                                              --focal_scaling $FOCAL_SCALING \
#                                              --mirror_shift $MIRR_SHIFT \

# # render
# python render.py \
#     -m $OUTPUT_DIR \
#     -s $DATASET_DIR \
#     --iteration $ITERS_NUM \
#     --camera_model FISHEYE \
#     --render_model $RENDER_MODEL \
#     --distortion_scaling $DIST_SCALING \
#     --focal_scaling $FOCAL_SCALING \
#     --mirror_shift $MIRR_SHIFT \
#     --skip_train \
#     --mask_path $DATASET_DIR$FOVMAP_DIR_EVAL$TEST_MASK_FN \
#     --raymap_path "$DATASET_DIR"raymap_fisheye.npy \
#     --sample_step $STEP_EVAL --fov_mod $FOVMOD_EVAL \
#     --train_test_exp \

# # wrap back to origianal space
# echo "Ground truth (kb) remapping from FoVMap"
# python extract_kb.py --path $DATASET_DIR \
#                     --src $OUTPUT_DIR/test/ours_$ITERS_NUM/gt \
#                     --dst $OUTPUT_DIR/test/ours_$ITERS_NUM/gt_remap \
#                     --step $STEP_EVAL --fov_mod $FOVMOD_EVAL --gridmap_restrict

# python extract_kb.py --path $DATASET_DIR \
#                      --src $OUTPUT_DIR/test/ours_$ITERS_NUM/renders \
#                      --dst $OUTPUT_DIR/test/ours_$ITERS_NUM/renders_remap \
#                      --step $STEP_EVAL --fov_mod $FOVMOD_EVAL --gridmap_restrict

# evaluation
python metrics.py \
    -m $OUTPUT_DIR"/test" \
    --iters $ITERS_NUM \
    --use_remap \
    --custom_mask ../../ablation/omni-gvr-scannetpp/output_fov/scannetpp/$SCENE_ID/test/ours_30000/renders_remap/mask.png \
    --reverse_mask \
    --block_mask \
    # --custom_gt /home/scannetpp_ever_gt/dslr/$SCENE_ID/test/ours_30000/gt \

# # evaluation
# python metrics.py \
#     -m /home/Fisheye-GS/output_scannetpp_fs_gt/scannetpp_fs/scannetpp/dslr/$SCENE_ID"/test" \
#     --use_remap \
#     --iters 30000 \
#     --custom_gt /home/scannetpp_ever_gt/dslr/$SCENE_ID/test/ours_30000/gt \
#     --custom_mask ../../ablation/omni-gvr-scannetpp/output_fov_updated/scannetpp/$SCENE_ID/test/ours_30000/renders_remap/mask.png \
#     --reverse_mask \
#     --block_mask \

# # evaluation
# python metrics.py \
#     -m /home/3dgrut/runs/$SCENE_ID"_3dgut"/dslr-1604_064548 \
#     --use_remap \
#     --iters 30000 \
#     --custom_gt /home/scannetpp_ever_gt/dslr/$SCENE_ID/test/ours_30000/gt \
#     --custom_mask ../../ablation/omni-gvr-scannetpp/output_fov_updated/scannetpp/$SCENE_ID/test/ours_30000/renders_remap/mask.png \
#     --reverse_mask \
#     --block_mask \

# # evaluation
# python metrics.py \
#     -m /home/scannetpp_ever_gt/dslr/$SCENE_ID"/test" \
#     --iters 30000 \
#     --custom_gt /home/scannetpp_ever_gt/dslr/$SCENE_ID/test/ours_30000/gt \
#     --custom_mask ../../ablation/omni-gvr-scannetpp/output_fov_updated/scannetpp/$SCENE_ID/test/ours_30000/renders_remap/mask.png \
#     --reverse_mask \
#     --block_mask \
