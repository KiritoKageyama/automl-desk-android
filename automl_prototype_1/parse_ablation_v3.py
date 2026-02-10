
import re
import os

def parse_results(filename):
    # Try UTF-16 first (PowerShell default), then UTF-8
    encodings = ['utf-16', 'utf-8']
    lines = []
    
    for enc in encodings:
        try:
            with open(filename, 'r', encoding=enc) as f:
                content = f.read()
                if "Full" in content or "NoProxy" in content:
                    lines = content.splitlines()
                    break
        except:
            continue
            
    if not lines:
        print("Could not read file with known encodings or file is empty.")
        return []

    results = []
    
    for line in lines:
        line = line.strip()
        parts = line.split()
        
        # Minimum parts: Algo, Clf, Case, Acc, F1, Feats, Time = 7 parts
        if len(parts) < 7:
            continue
            
        # Check if the last part is an integer (Time)
        # Handle potential invisible chars
        time_str = parts[-1].replace('\x00', '')
        if not time_str.isdigit():
            continue
            
        # Check if the 2nd to last looks like "N/M"
        if '/' not in parts[-2]:
            continue
            
        known_cases = ["Full", "NoProxy", "NoGuide"]
        
        case_candidate = parts[-5]
        # Clean potential nulls
        case_candidate = case_candidate.replace('\x00', '')
        
        if case_candidate not in known_cases:
            continue
            
        algo = parts[0].replace('\x00', '')
        classifier = " ".join(parts[1:-5]).replace('\x00', '')
        case = case_candidate
        
        try:
            acc = float(parts[-4].replace('\x00', ''))
            f1 = float(parts[-3].replace('\x00', ''))
            time = int(time_str)
            feats_str = parts[-2].replace('\x00', '')
            sel = int(feats_str.split('/')[0])
        except:
            continue

        results.append({
            'Algorithm': algo,
            'Classifier': classifier,
            'Case': case,
            'Accuracy': acc,
            'F1': f1,
            'Selected': sel,
            'Time': time
        })

    return results

def generate_markdown(results):
    if not results:
        print("No valid data rows found found.")
        return

    case_order = {'Full': 0, 'NoProxy': 1, 'NoGuide': 2}
    
    results.sort(key=lambda x: (x['Algorithm'], x['Classifier'], case_order.get(x['Case'], 99)))
    
    print("| Algorithm | Classifier | Ablation Case | Accuracy | F1 Score | Selected Feats | Time (ms) |")
    print("|-----------|------------|---------------|----------|----------|----------------|-----------|")
    
    for r in results:
        c_disp = r['Case']
        if c_disp == 'Full': c_disp = 'Full System'
        if c_disp == 'NoProxy': c_disp = 'No Fast-Proxy'
        if c_disp == 'NoGuide': c_disp = 'No Guidance'
        
        print(f"| {r['Algorithm']} | {r['Classifier']} | {c_disp} | {r['Accuracy']:.4f} | {r['F1']:.4f} | {r['Selected']} | {r['Time']} |")

if __name__ == "__main__":
    data = parse_results('comprehensive_ablation_results_v3.txt')
    generate_markdown(data)
