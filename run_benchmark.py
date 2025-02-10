import os
import subprocess
from datetime import datetime

# Change permissions recursively
subprocess.run(["sudo", "chmod", "-R", "777", "./datasets/"])
file_path = "f3rm_robot/args.py"
time_stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
new_model_name = "dex-f3rm_" + time_stamp
output_path = f"datasets/eyeinhand_nerf1/benchmark/{new_model_name}"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Open the Python file and modify the model name
with open(file_path, "r") as file:
    lines = file.readlines()

with open(file_path, "w") as file:
    for line in lines:
        if "model_name: str = Proto(" in line:
            line = f"    model_name: str = Proto(\"{new_model_name}\", help=\"Benchmark scenes\")\n"
        file.write(line)
os.popen(f"cp f3rm_robot/args.py {output_path}")

for i in range(1, 6):
    scene_path = f"datasets/eyeinhand_nerf1/img_ycb_scene_{i}"
    print(scene_path)

    subprocess.run([
        "f3rm-optimize", "--scene", f"f3rm_outputs/img_ycb_scene_{i}/f3rm/config.yml"
    ])