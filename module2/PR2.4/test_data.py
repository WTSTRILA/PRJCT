import pandas as pd

file_path = './data/annotated_data.csv'
df = pd.read_csv(file_path)


def get_expected_annotations(row):
    annotations = []
    if row['temperature'] > 100:
        annotations.append("Overheating")
    if row['pressure'] > 200:
        annotations.append("Pressure Issue")
    if row['vibration_x'] > 0.2 or row['vibration_y'] > 0.2 or row['vibration_z'] > 0.2:
        annotations.append("Vibration Issue")
    if row['oil_level'] < 5:
        annotations.append("Low Oil Level")
    if row['tablet_weight'] < 0.5 or row['tablet_weight'] > 1.5:
        annotations.append("Tablet Weight Issue")
    if row['tablet_thickness'] < 2 or row['tablet_thickness'] > 4:
        annotations.append("Tablet Thickness Issue")
    if row['tooling_wear'] > 0.1:
        annotations.append("Tooling Wear Issue")
    return set(annotations)


def check_annotations(df):
    correct = 0
    incorrect = 0

    for _, row in df.iterrows():
        expected_annotations = get_expected_annotations(row)
        actual_annotations = set(row['context'].split(',')) if pd.notna(row['context']) else set()

        if expected_annotations == actual_annotations:
            correct += 1
        else:
            incorrect += 1
            print(f"Incorrect annotation at index {row.name}:")
            print(f"Expected: {expected_annotations}")
            print(f"Actual: {actual_annotations}")

    print(f"Total correct annotations: {correct}")
    print(f"Total incorrect annotations: {incorrect}")


check_annotations(df)
