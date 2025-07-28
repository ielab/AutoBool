import os
import argparse

def compute_f_scores(recall, precision, beta):
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

def process_eval_file(filepath):
    original_lines = []
    updated_lines = []
    f1_list = []
    f3_list = []
    f5_list = []
    avg_recall = []
    avg_precision = []

    with open(filepath, 'r') as f:
        original_lines = f.readlines()

    changed = False

    for line in original_lines:
        parts = line.strip().split('\t')
        if len(parts) < 3:
            updated_lines.append(line)
            continue

        topic_id = parts[0]
        try:
            recall = float(parts[1])
            precision = float(parts[2])
        except ValueError:
            updated_lines.append(line)
            continue

        if topic_id == "AVG":
            continue  # skip old AVG line

        f1 = compute_f_scores(recall, precision, beta=1)
        f3 = compute_f_scores(recall, precision, beta=3)
        f5 = compute_f_scores(recall, precision, beta=5)
        f1_list.append(f1)
        f3_list.append(f3)
        f5_list.append(f5)
        avg_recall.append(recall)
        avg_precision.append(precision)

        new_line = f"{topic_id}\t{recall:.4f}\t{precision:.4f}\t{f1:.4f}\t{f3:.4f}\t{f5:.4f}\n"
        updated_lines.append(new_line)

    if avg_recall and avg_precision:
        avg_line = (
            f"AVG\t{sum(avg_recall)/len(avg_recall):.4f}"
            f"\t{sum(avg_precision)/len(avg_precision):.4f}"
            f"\t{sum(f1_list)/len(f1_list):.4f}"
            f"\t{sum(f3_list)/len(f3_list):.4f}"
            f"\t{sum(f5_list)/len(f5_list):.4f}\n"
        )
        updated_lines.append(avg_line)

    # Only write if there's a change
    if updated_lines != original_lines:
        with open(filepath, 'w') as f:
            f.writelines(updated_lines)
        print(f"Updated: {filepath}")
    else:
        print(f"No changes needed: {filepath}")


def compute_additional_f1_f3(input_folder):
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.eval'):
                filepath = os.path.join(root, file)
                process_eval_file(filepath)

def main():
    parser = argparse.ArgumentParser(description="Process .eval files and compute F1, F3, F5 scores.")
    parser.add_argument("input_folder", help="Path to the input folder containing .eval files")
    args = parser.parse_args()

    compute_additional_f1_f3(args.input_folder)

if __name__ == "__main__":
    main()
