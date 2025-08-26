import pandas as pd
import json
from collections import defaultdict

def read_data_from_csv(read_file, write_file):
    """
    Process custom JSON data format and convert to pyKT data.txt format.
    
    pyKT data.txt format (6 lines per student):
    Line 0: user_id,sequence_length
    Line 1: question_ids (comma-separated) or NA
    Line 2: concept_ids (comma-separated) 
    Line 3: responses (comma-separated)
    Line 4: timestamps (comma-separated) or NA
    Line 5: usetimes (comma-separated) or NA
    """
    print(f"Processing custom data from: {read_file}")
    
    # Load JSON data from the train_valid.csv file (which is actually our converted data)
    df = pd.read_csv(read_file)
    
    all_lines = []
    
    for _, row in df.iterrows():
        user_id = row['uid']
        concepts = row['concepts'].split(',')
        responses = row['responses'].split(',')
        
        seq_len = len(concepts)
    
        # Line 0: user_id,sequence_length
        all_lines.append(f"{user_id},{seq_len}")
        
        # Line 1: question_ids (we don't have questions, so use concepts as questions)
        all_lines.append(','.join(concepts))
        
        # Line 2: concept_ids 
        all_lines.append(','.join(concepts))
        
        # Line 3: responses
        all_lines.append(','.join(responses))
        
        # Line 4: timestamps (we don't have real timestamps, use sequential numbers)
        timestamps = [str(i) for i in range(seq_len)]
        all_lines.append(','.join(timestamps))
        
        # Line 5: usetimes (we don't have use times, use NA)
        all_lines.append('NA')
    
    # Write to data.txt
    with open(write_file, 'w') as f:
        f.write('\n'.join(all_lines))
    
    print(f"Processed {len(df)} students with {sum(len(row['concepts'].split(',')) for _, row in df.iterrows())} interactions")
    print(f"Written to: {write_file}")
    
    return write_file