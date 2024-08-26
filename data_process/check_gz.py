import gzip

def load_text(file_path):
    with gzip.open(file_path, 'rt') as f:
        data = f.readlines()
        print(len(data))
        
        
load_text("data/bio/instruction_biomed/part_0.json.gz")