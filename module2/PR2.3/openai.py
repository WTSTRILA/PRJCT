import json
import pandas as pd
from typing import Dict
from retry import retry
from openai import OpenAI

api_key = ""
openai.api_key = api_key

prompt = """
Сгенерируй датасет в 50 строк который будет использоваться для обучение модели предсказания обслуживания технологического оборудования - таблетпрес, в датасете должны быть следующие столбцы:
timestamp, press_id, temperature, pressure, vibration_x, vibration_y, vibration_z, rotation_speed, motor_current, oil_level, humidity, ambient_temperature, noise_level, tablet_weight, tablet_thickness, tablet_hardness, tooling_wear, tablet_output, production_rate

Представь результат в формате JSON в следующем виде :
{
    "timestamp": "****-**-** **:**:**",
    "press_id": *,
    "temperature": **.*,
    "pressure": ***.*,
    "vibration_x": *.**,
    "vibration_y": *.**,
    "vibration_z": *.**,
    "rotation_speed": ****,
    "motor_current": **.*,
    "oil_level": *.*,
    "humidity": **,
    "ambient_temperature": **.*,
    "noise_level": **,
    "tablet_weight": *.***,
    "tablet_thickness": *.*,
    "tablet_hardness": **.*,
    "tooling_wear": *.**,
    "tablet_output": ***,
    "production_rate": ***,
}

Сделай шаг между записями в 6 часов 
"""

@retry(tries=3, delay=1)
def generate_synthetic_example() -> Dict[str, str]:
    client = OpenAI()
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are technical engineer expert.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        model="gpt-4o",
        response_format={"type": "json_object"},
        temperature=1,
    )
    sample = json.loads(chat_completion.choices[0].message.content)
    return sample


def create_synthetic_dataset(num_samples: int = 10):
    samples = [generate_synthetic_example() for _ in range(num_samples)]

    df = pd.DataFrame(samples)
    df.to_csv("data.csv", index=False)
    print("Synthetic data saved" )

if __name__ == "__main__":
    create_synthetic_dataset()
