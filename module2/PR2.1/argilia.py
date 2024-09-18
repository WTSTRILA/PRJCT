import pandas as pd
import argilla as rg

client = rg.Argilla(api_url="http://localhost:6900", api_key="admin.apikey")


def create_sensor_dataset(csv_file_path):
    guidelines = """
    Please examine the provided sensor data and annotate the data based on the type of equipment malfunction. 
    """

    dataset_name = "sensor_data_v6"
    settings = rg.Settings(
        guidelines=guidelines,
        fields=[
            rg.TextField(
                name="query",
                title="Sensor Data",
                use_markdown=False,
            ),
            rg.TextField(
                name="context",
                title="Context",
                use_markdown=True,
            ),
        ],
        questions=[
            rg.TextQuestion(
                name="annotation",
                title="Equipment Malfunction Annotation",
                description="Annotate the data based on the type of malfunction detected.",
                required=True,
                use_markdown=True,
            )
        ],
    )

    try:
        dataset = client.get_dataset(name=dataset_name)
        print(f"Dataset '{dataset_name}' already exists.")
    except Exception:
        dataset = rg.Dataset(
            name=dataset_name,
            workspace="admin",
            settings=settings,
            client=client,
        )
        dataset.create()
        print(f"Dataset '{dataset_name}' created.")

    df = pd.read_csv(csv_file_path)
    records = []
    for idx, row in df.iterrows():
        query_data = f"Timestamp: {row['timestamp']}, Press ID: {row['press_id']}, Temperature: {row['temperature']}, Pressure: {row['pressure']}, Vibration X: {row['vibration_x']}, Vibration Y: {row['vibration_y']}, Vibration Z: {row['vibration_z']}, Rotation Speed: {row['rotation_speed']}, Motor Current: {row['motor_current']}, Oil Level: {row['oil_level']}, Humidity: {row['humidity']}, Ambient Temperature: {row['ambient_temperature']}, Noise Level: {row['noise_level']}, Tablet Weight: {row['tablet_weight']}, Tablet Thickness: {row['tablet_thickness']}, Tablet Hardness: {row['tablet_hardness']}, Tooling Wear: {row['tooling_wear']}, Tablet Output: {row['tablet_output']}, Production Rate: {row['production_rate']}"

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

        context_data = "; ".join(annotations)

        record = rg.Record(
            fields={
                "query": query_data,
                "context": context_data
            }
        )
        records.append(record)

        df.at[idx, 'context'] = context_data

    dataset.records.log(records)

    annotated_csv_file_path = ".data/annotated_data.csv"
    df.to_csv(annotated_csv_file_path, index=False)
    print(f"Annotated dataset saved to {annotated_csv_file_path}")


if __name__ == "__main__":
    csv_file_path = ".data/data.csv"
    create_sensor_dataset(csv_file_path)
