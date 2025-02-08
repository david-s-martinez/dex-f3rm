sudo chmod -R 777 ./datasets/
is_new="false"

for i in $(seq 1 5);
do
    echo datasets/eyeinhand_nerf1/img_ycb_scene_$i
    if [[ "$is_new" == "true" ]]; then
        rm -rf datasets/eyeinhand_nerf1/img_ycb_scene_$i/f3rm_clip_features.pt 
        rm -rf f3rm_outputs/img_ycb_scene_$i/f3rm
        ns-train f3rm --max-num-iterations 5000 --output_dir f3rm_outputs --experiment_name img_ycb_scene_$i --timestamp '' nerfstudio-data --data ./datasets/eyeinhand_nerf1/img_ycb_scene_$i --orientation-method none --auto-scale-poses True
    fi
    f3rm-optimize --scene f3rm_outputs/img_ycb_scene_$i/f3rm/config.yml
done