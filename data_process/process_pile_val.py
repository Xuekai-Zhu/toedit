import json
import os
from collections import defaultdict

def split_json_by_pile_set_name(input_file, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        all_lines = f_in.readlines()
        
    data = [json.loads(i) for i in all_lines]

    split_data = defaultdict(list)
    for item in data:
        pile_set_name = item.get("meta", {}).get("pile_set_name")
        if pile_set_name:
            split_data[pile_set_name].append(item)
    
    # Write each category to a separate file
    for pile_set_name, items in split_data.items():
        
        output_file = os.path.join(output_dir, f"{pile_set_name.replace(' ', '_')}.json")
        
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for j in items:
                in_str = json.dumps(j) + "\n"
                f_out.write(in_str)
                
        print(f"Saved {len(items)} items to {output_file}")
        

# Example usage
split_json_by_pile_set_name("data/pile-val/val.jsonl", "data/pile-val")

