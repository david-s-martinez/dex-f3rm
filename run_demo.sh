sudo chmod -R 777 ./datasets/
# rm -rf datasets/eyeinhand_nerf1/img_a_demo_scene/colmap datasets/eyeinhand_nerf1/img_a_demo_scene/images_2 datasets/eyeinhand_nerf1/img_a_demo_scene/images_4 datasets/eyeinhand_nerf1/img_a_demo_scene/images_8
# ns-process-data images --data datasets/eyeinhand_nerf1/img_a_demo_scene/images --output-dir datasets/eyeinhand_nerf1/img_a_demo_scene
# echo "/workspaces/f3rm/datasets/eyeinhand_nerf1/img_a_demo_scene" > "/workspaces/f3rm/datasets/calibration_dir.txt"
# python3 f3rm/scripts/colmap_to_world.py
rm -rf datasets/eyeinhand_nerf1/img_a_demo_scene/f3rm_clip_features.pt 
rm -rf f3rm_outputs/a_demo_scene/f3rm
ns-train f3rm --max-num-iterations 5000 --output_dir f3rm_outputs --experiment_name a_demo_scene --timestamp '' nerfstudio-data --data ./datasets/eyeinhand_nerf1/img_a_demo_scene --orientation-method none --auto-scale-poses True
f3rm-optimize --scene f3rm_outputs/a_demo_scene/f3rm/config.yml