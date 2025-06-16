
import pandas as pd
import random

def generate_sample():
    sleep = round(random.uniform(5.0, 9.0), 1)
    wake_time = random.randint(4, 10)
    stress = random.randint(1, 10)
    water = round(random.uniform(1.5, 4.0), 1)
    meal = random.randint(1, 10)
    exercise = random.randint(0, 7)
    region = random.choice(["hot", "cold", "humid"])
    temp_feel = random.choice(["cold", "warm"])
    
    # Dosha logic (simplified)
    vata = 50 + (wake_time - 7) * 5 + (stress - 5) * 2
    pitta = 50 + (stress * 3) - (water * 5)
    kapha = 50 + (sleep - 7) * 5 + (meal - 5) * 2 - (exercise * 2)

    # Clamp between 0–100
    vata = min(max(vata, 0), 100)
    pitta = min(max(pitta, 0), 100)
    kapha = min(max(kapha, 0), 100)

    return [sleep, wake_time, stress, water, meal, exercise, region, temp_feel, vata, pitta, kapha]

data = []
for _ in range(150):
    data.append(generate_sample())

columns = ["sleep", "wake_time", "stress", "water", "meal", "exercise", "region", "temp_feel", "vata", "pitta", "kapha"]
df = pd.DataFrame(data, columns=columns)

df.to_csv("data/dosha_data.csv", index=False)
print("✅ Data generated: dosha_data.csv")
