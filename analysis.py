import os
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def load_predictions(symbols):
    """Load predictions from output files into a dictionary."""
    predictions_data = {}
    for symbol in symbols:
        file_path = f"{symbol}_predictions.txt"
        if not os.path.exists(file_path):
            print(f"File {file_path} not found.")
            continue

        data = []
        with open(file_path, "r") as file:
            file.readline()  # Skip header
            for line in file:
                parts = line.strip().split("\t")
                tick_index = int(parts[0])
                prediction = int(parts[1])
                confidence = float(parts[2])
                data.append((tick_index, prediction, confidence))

        predictions_data[symbol] = pd.DataFrame(data, columns=["TickIndex", "Prediction", "Confidence"])
    return predictions_data

def load_original_data(symbols, data_dir="./test-comp/"):
    """Load original data from the text files."""
    original_data = {}
    for symbol in symbols:
        file_path = os.path.join(data_dir, f"{symbol}_data.txt")
        if not os.path.exists(file_path):
            print(f"File {file_path} not found.")
            continue

        df = pd.read_csv(file_path, delimiter=r"\s+", engine='python')
        original_data[symbol] = df
    return original_data

def align_and_evaluate(predictions_data, original_data, symbol, output_dir):
    """Align predictions with the original data and evaluate performance."""
    predictions_df = predictions_data[symbol]
    original_df = original_data[symbol]

    # Merge on TickIndex to align predictions with actual labels
    merged_df = pd.merge(predictions_df, original_df, left_on="TickIndex", right_index=True, how="inner")
    merged_df.rename(columns={"Label": "Actual"}, inplace=True)

    # Generate overall classification report
    overall_report = classification_report(merged_df["Actual"], merged_df["Prediction"], digits=4)
    
    # Calculate overall accuracy, precision, recall, and F1-score
    overall_accuracy = accuracy_score(merged_df["Actual"], merged_df["Prediction"])
    overall_precision = precision_score(merged_df["Actual"], merged_df["Prediction"], average='weighted', zero_division=0)
    overall_recall = recall_score(merged_df["Actual"], merged_df["Prediction"], average='weighted', zero_division=0)
    overall_f1 = f1_score(merged_df["Actual"], merged_df["Prediction"], average='weighted', zero_division=0)
    
    classification_results = [
        f"Overall Classification Report for {symbol}:\n{overall_report}",
        f"Accuracy: {overall_accuracy:.4f}",
        f"Precision: {overall_precision:.4f}",
        f"Recall: {overall_recall:.4f}",
        f"F1 Score: {overall_f1:.4f}"
    ]

    # Analyze by confidence levels
    confidence_ranges = [(i, i + 10) for i in range(0, 100, 10)]
    for start, end in confidence_ranges:
        range_df = merged_df[(merged_df['Confidence'] >= start / 100) & (merged_df['Confidence'] < end / 100)]
        if range_df.empty:
            continue

        accuracy = accuracy_score(range_df["Actual"], range_df["Prediction"])
        precision = precision_score(range_df["Actual"], range_df["Prediction"], average='weighted', zero_division=0)
        recall = recall_score(range_df["Actual"], range_df["Prediction"], average='weighted', zero_division=0)
        f1 = f1_score(range_df["Actual"], range_df["Prediction"], average='weighted', zero_division=0)
        report = classification_report(range_df["Actual"], range_df["Prediction"], digits=4)

        classification_results.append(f"\nClassification Report for Confidence Range {start}-{end}%:")
        classification_results.append(f"Accuracy: {accuracy:.4f}")
        classification_results.append(f"Precision: {precision:.4f}")
        classification_results.append(f"Recall: {recall:.4f}")
        classification_results.append(f"F1 Score: {f1:.4f}")
        classification_results.append(report)

    # Save the classification report
    classification_report_path = os.path.join(output_dir, f"{symbol}_classification_analysis.txt")
    with open(classification_report_path, "w") as output_file:
        output_file.write("\n".join(classification_results))
    print(f"Classification analysis saved to {classification_report_path}")

    # Analyze confidence level distribution for each class
    confidence_distribution_results = [f"Confidence Level Distribution for {symbol}:"]
    for pred_class in range(3):
        confidence_distribution_results.append(f"\nClass {pred_class}:")
        class_df = merged_df[merged_df["Prediction"] == pred_class]
        class_total = len(class_df)
        
        for start, end in confidence_ranges:
            range_df = class_df[(class_df['Confidence'] >= start / 100) & (class_df['Confidence'] < end / 100)]
            count_in_range = len(range_df)
            percentage = (count_in_range / class_total) * 100 if class_total > 0 else 0
            confidence_distribution_results.append(f"{start}-{end}%: {count_in_range} ({percentage:.2f}%)")
    
    # Save the confidence distribution report
    confidence_report_path = os.path.join(output_dir, f"{symbol}_confidence_distribution.txt")
    with open(confidence_report_path, "w") as output_file:
        output_file.write("\n".join(confidence_distribution_results))
    print(f"Confidence distribution saved to {confidence_report_path}")


def main():
    symbols = ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCHF"]

    # Load predictions and original data
    predictions_data = load_predictions(symbols)
    original_data = load_original_data(symbols)

    # Create output directory for analysis reports
    output_dir = "./5_epoch_3/"
    os.makedirs(output_dir, exist_ok=True)

    # Perform alignment and analysis for each symbol
    for symbol in symbols:
        if symbol in predictions_data and symbol in original_data:
            align_and_evaluate(predictions_data, original_data, symbol, output_dir)

if __name__ == "__main__":
    main()
