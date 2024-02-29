import os
from utils import generate_lines_testcases, retrieve_cmd_args

if __name__ == "__main__":
    cfgs = retrieve_cmd_args()

    output_dir = os.path.join(cfgs["output_dir"], cfgs["task"], "testcases/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        raise RuntimeError("Output directory already exists! Remove this line to overwrite the original testcases.")

    if cfgs["task"] == "topics":
        pass
    else:
        generate_lines_testcases(cfgs, output_dir)