
import re

def parse_results(filename):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    results = []
    
    # We are looking for lines that end with a number (Time) and have the Case keyword in them etc.
    # Pattern: Algo(1) Classifier(N) Case(1) Acc(1) F1(1) Feats(1) Time(1)
    # Strategy: Split line by whitespace.
    # Check if the line matches the structure.
    
    for line in lines:
        line = line.strip()
        parts = line.split()
        
        # Minimum parts: Algo, Clf, Case, Acc, F1, Feats, Time = 7 parts
        if len(parts) < 7:
            continue
            
        # Check if the last part is an integer (Time)
        if not parts[-1].isdigit():
            continue
            
        # Check if the 2nd to last looks like "N/M"
        if '/' not in parts[-2]:
            continue
            
        # Check if the 5th to last is a known Case
        known_cases = ["Full", "NoProxy", "NoGuide"]
        
        # We need to find which part is the Case.
        # It should be at index -5?
        # Let's count from right:
        # -1: Time
        # -2: Feats (N/M)
        # -3: F1
        # -4: Acc
        # -5: Case?
        
        case_candidate = parts[-5]
        if case_candidate not in known_cases:
            continue
            
        # Algorithm is parts[0]
        algo = parts[0]
        
        # Classifier is everything between parts[0] and parts[-5]
        classifier = " ".join(parts[1:-5])
        
        case = case_candidate
        acc = float(parts[-4])
        f1 = float(parts[-3])
        time = int(parts[-1])
        feats_str = parts[-2] # "22/31"
        try:
            sel = int(feats_str.split('/')[0])
        except:
            sel = 0

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
    # Sort
    case_order = {'Full': 0, 'NoProxy': 1, 'NoGuide': 2}
    
    # Sort by Algorithm, then Classifier, then Case
    results.sort(key=lambda x: (x['Algorithm'], x['Classifier'], case_order.get(x['Case'], 99)))
    
    print("| Algorithm | Classifier | Ablation Case | Accuracy | F1 Score | Selected Feats | Time (ms) |")
    print("|-----------|------------|---------------|----------|----------|----------------|-----------|")
    
    for r in results:
        # Rename Cases for report if desired
        c_disp = r['Case']
        if c_disp == 'Full': c_disp = 'Full System'
        if c_disp == 'NoProxy': c_disp = 'No Fast-Proxy'
        if c_disp == 'NoGuide': c_disp = 'No Guidance'
        
        print(f"| {r['Algorithm']} | {r['Classifier']} | {c_disp} | {r['Accuracy']:.4f} | {r['F1']:.4f} | {r['Selected']} | {r['Time']} |")

if __name__ == "__main__":
    data = parse_results('comprehensive_ablation_results_v3.txt')
    generate_markdown(data)
