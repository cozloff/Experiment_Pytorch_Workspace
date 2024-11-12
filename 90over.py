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

def evaluate_high_confidence_predictions(predictions_data, original_data, symbols, output_dir):
    """Evaluate performance only for predictions with confidence > 90%."""
    overall_actuals = []
    overall_predictions = []

    for symbol in symbols:
        if symbol not in predictions_data or symbol not in original_data:
            print(f"Data for {symbol} is missing.")
            continue

        predictions_df = predictions_data[symbol]
        original_df = original_data[symbol]

        # Merge predictions with original labels
        merged_df = pd.merge(predictions_df, original_df, left_on="TickIndex", right_index=True, how="inner")
        merged_df.rename(columns={"Label": "Actual"}, inplace=True)

        # Filter predictions with confidence > 90%
        high_confidence_df = merged_df[merged_df["Confidence"] > 0.9]
        if high_confidence_df.empty:
            print(f"No high-confidence predictions for {symbol}.")
            continue

        # Generate classification report for each symbol
        symbol_actuals = high_confidence_df["Actual"].tolist()
        symbol_predictions = high_confidence_df["Prediction"].tolist()
        report = classification_report(symbol_actuals, symbol_predictions, digits=4)

        accuracy = accuracy_score(symbol_actuals, symbol_predictions)
        precision = precision_score(symbol_actuals, symbol_predictions, average='weighted', zero_division=0)
        recall = recall_score(symbol_actuals, symbol_predictions, average='weighted', zero_division=0)
        f1 = f1_score(symbol_actuals, symbol_predictions, average='weighted', zero_division=0)

        # Save the classification report for each pair
        report_lines = [
            f"Classification Report for {symbol} (Confidence > 90%):",
            report,
            f"Accuracy: {accuracy:.4f}",
            f"Precision: {precision:.4f}",
            f"Recall: {recall:.4f}",
            f"F1 Score: {f1:.4f}"
        ]

        report_path = os.path.join(output_dir, f"{symbol}_classification_report.txt")
        with open(report_path, "w") as report_file:
            report_file.write("\n".join(report_lines))
        
        print(f"Classification report for {symbol} saved to {report_path}")

        # Accumulate data for overall report
        overall_actuals.extend(symbol_actuals)
        overall_predictions.extend(symbol_predictions)

    # Generate overall classification report for all pairs combined
    if len(overall_actuals) > 0:
        overall_report = classification_report(overall_actuals, overall_predictions, digits=4)
        overall_accuracy = accuracy_score(overall_actuals, overall_predictions)
        overall_precision = precision_score(overall_actuals, overall_predictions, average='weighted', zero_division=0)
        overall_recall = recall_score(overall_actuals, overall_predictions, average='weighted', zero_division=0)
        overall_f1 = f1_score(overall_actuals, overall_predictions, average='weighted', zero_division=0)

        overall_report_lines = [
            "Accumulated Classification Report (Confidence > 90% for all pairs):",
            overall_report,
            f"Accuracy: {overall_accuracy:.4f}",
            f"Precision: {overall_precision:.4f}",
            f"Recall: {overall_recall:.4f}",
            f"F1 Score: {overall_f1:.4f}"
        ]

        overall_report_path = os.path.join(output_dir, "overall_high_confidence_report.txt")
        with open(overall_report_path, "w") as report_file:
            report_file.write("\n".join(overall_report_lines))
        
        print(f"Overall high-confidence classification report saved to {overall_report_path}")

def main():
    symbols = ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCHF"]

    # Load predictions and original data
    predictions_data = load_predictions(symbols)
    original_data = load_original_data(symbols)

    # Create output directory for the report
    output_dir = "./high_confidence_90_analysis/"
    os.makedirs(output_dir, exist_ok=True)

    # Perform analysis on high-confidence predictions
    evaluate_high_confidence_predictions(predictions_data, original_data, symbols, output_dir)

if __name__ == "__main__":
    main()
