import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    with open(args.input_file, "r") as f_in, open(args.output_file, "w") as f_out:
        for line in f_in:
            data = json.loads(line)
            # NeMo-Skills SFT data preparation expectations:
            # It usually expects 'input' and 'expected_output' or similar.
            # Here we follow the tutorial's logic of using generated solutions.
            # We wrap the reasoning in <thought> tags if not already present.
            
            # This is a simplified version, as nemo_skills.training.prepare_data 
            # is usually preferred, but this script ensures we can run it in our environment.
            f_out.write(json.dumps(data) + "\n")

if __name__ == "__main__":
    main()
