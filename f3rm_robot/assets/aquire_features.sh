echo "Getting features for Teddy Bear..."
# python3 f3rm_robot/examples/generate_task.py --scene f3rm_outputs/teddy_bear/f3rm/config.yml --demo_fname scene_demo_teddy_bear.json --save --disable_visualize
python3 f3rm_robot/examples/generate_task.py --scene f3rm_outputs/teddy_bear/f3rm/config.yml --demo_fname scene_demo_teddy_bear_head.json --save --disable_visualize
python3 f3rm_robot/examples/generate_task.py --scene f3rm_outputs/teddy_bear/f3rm/config.yml --demo_fname scene_demo_teddy_bear_ears.json --save --disable_visualize
python3 f3rm_robot/examples/generate_task.py --scene f3rm_outputs/teddy_bear/f3rm/config.yml --demo_fname scene_demo_teddy_bear_arms.json --save --disable_visualize
python3 f3rm_robot/examples/generate_task.py --scene f3rm_outputs/teddy_bear_laying/f3rm/config.yml --demo_fname scene_demo_laying_teddy_bear.json --save --disable_visualize
echo "Getting features for Bowl..."
python3 f3rm_robot/examples/generate_task.py --scene f3rm_outputs/bowl_cup/f3rm/config.yml --demo_fname scene_demo_bowl.json --save --disable_visualize
echo "Getting features for Mug..."
# python3 f3rm_robot/examples/generate_task.py --scene f3rm_outputs/bowl_cup/f3rm/config.yml --demo_fname scene_demo_mug.json --save --disable_visualize
python3 f3rm_robot/examples/generate_task.py --scene f3rm_outputs/bowl_cup/f3rm/config.yml --demo_fname scene_demo_mug_handle.json --save --disable_visualize
python3 f3rm_robot/examples/generate_task.py --scene f3rm_outputs/bowl_cup/f3rm/config.yml --demo_fname scene_demo_mug_body.json --save --disable_visualize
echo "Getting features for Black Headphones..."
python3 f3rm_robot/examples/generate_task.py --scene f3rm_outputs/headphones_crackers/f3rm/config.yml --demo_fname scene_demo_black_headphones.json --save --disable_visualize
echo "Getting features for Crackers Box..."
python3 f3rm_robot/examples/generate_task.py --scene f3rm_outputs/headphones_crackers/f3rm/config.yml --demo_fname scene_demo_crackers_box.json --save --disable_visualize
echo "Getting features for Sweep..."
python3 f3rm_robot/examples/generate_task.py --scene f3rm_outputs/mentos_sweep/f3rm/config.yml --demo_fname scene_demo_gray_sweep.json --save --disable_visualize
echo "Getting features for Mentos..."
python3 f3rm_robot/examples/generate_task.py --scene f3rm_outputs/mentos_sweep/f3rm/config.yml --demo_fname scene_demo_mentos_gum.json --save --disable_visualize
echo "Getting features for Black Cube..."
python3 f3rm_robot/examples/generate_task.py --scene f3rm_outputs/cubess/f3rm/config.yml --demo_fname scene_demo_black_foam_cube.json --save --disable_visualize
echo "Getting features for Rubiks Cube..."
python3 f3rm_robot/examples/generate_task.py --scene f3rm_outputs/cubess/f3rm/config.yml --demo_fname scene_demo_rubiks_cube.json --save --disable_visualize
echo "Getting features for Peach..."
python3 f3rm_robot/examples/generate_task.py --scene f3rm_outputs/cubess/f3rm/config.yml --demo_fname scene_demo_peach.json --save --disable_visualize
