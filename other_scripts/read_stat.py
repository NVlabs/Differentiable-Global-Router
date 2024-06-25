import os
import re
import sys
benchmark_path=sys.argv[1]

def extract_scores_from_splitting(filename, data, method):
    with open(filename, 'r') as f:
        lines = f.readlines()

        scores = {"data": data, "method": method}
        
        sum_guides_and_tracks_obedience = 0
        sum_design_rule_violations = 0
        
        # Flags to indicate which section we're in
        in_guides_and_tracks_obedience = False
        in_design_rule_violations = False
        
        for line in lines:
            # Summing up scores for the respective sections
            if in_guides_and_tracks_obedience and "|" in line and 'Design Rule Violations' not in line:
                sum_guides_and_tracks_obedience += float(line.split('|')[5].strip())
            elif in_design_rule_violations and "|" in line and 'Connectivity' not in line:
                sum_design_rule_violations += float(line.split('|')[5].strip())
            if 'Total wire length' in line:
                scores["total_wire_length_score"] = float(line.split('|')[5].strip())
            elif 'Total via count' in line or 'Total SCut via count' in line:
                scores["total_via_count_score"] = float(line.split('|')[5].strip())
            elif 'Total Score' in line:
                scores["total_score"] = float(line.split('|')[2].strip())
            elif 'Guides and tracks Obedience' in line:
                in_guides_and_tracks_obedience = True
                in_design_rule_violations = False  # Resetting other flags
            elif 'Design Rule Violations' in line:
                in_design_rule_violations = True
                in_guides_and_tracks_obedience = False  # Resetting other flags
            elif 'Connectivity' in line:
                in_design_rule_violations = False  # Ending the Design Rule Violations section
        scores["guides_and_tracks_obedience_score"] = sum_guides_and_tracks_obedience
        scores["design_rule_violation_score"] = sum_design_rule_violations

        return scores


def process_logs():
    results = []
    for root, dirs, files in os.walk(benchmark_path):
        for file in files:
            if re.match(r".*_.+\.final\.log$", file):
                filepath = os.path.join(root, file)
                
                # Extracting data and method from the path
                data = os.path.basename(root)
                method = file.split('_')[0]

                scores = extract_scores_from_splitting(filepath, data, method)
                results.append(scores)
    return results

if __name__ == "__main__":
    scores_list = process_logs()
    for scores in scores_list:
        try:
            print(f"Data: {scores['data']}, Method: {scores['method']}")
            print(f"Total Wire Length Score: {scores['total_wire_length_score']}")
            print(f"Total Via Count Score: {scores['total_via_count_score']}")
            print(f"Guides and Tracks Obedience Score: {scores['guides_and_tracks_obedience_score']}")
            print(f"Design Rule Violation Score: {scores['design_rule_violation_score']}")
            print(f"Total Score: {scores['total_score']}")
            print("-" * 50)  # separator for clarity
        except KeyError:
            print(f"Data: {scores['data']}, Method: {scores['method']}")
            print("Some scores are missing")
            print("-" * 50)
