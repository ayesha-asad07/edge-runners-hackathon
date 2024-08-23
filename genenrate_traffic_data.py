import openai
import random
import json

# Replace with your Llama API key
openai.api_key = "LA-API-KEY"

def generate_synthetic_traffic_data(num_samples=100):
    traffic_data = []
    for _ in range(num_samples):
        # Generate random data for normal and unusual traffic patterns
        sample = {
            "timestamp": random.randint(1609459200, 1704067200),  # Unix timestamp between 2021-2034
            "traffic_volume": random.randint(500, 10000),         # Random traffic volume
            "spike_detected": random.choice([0, 1]),              # 0 for normal, 1 for spike
            "unauthorized_access": random.choice([0, 1]),         # 0 for no, 1 for yes
        }
        traffic_data.append(sample)

    return traffic_data

def save_data_to_json(data, filename="traffic_data.json"):
    with open(filename, "w") as outfile:
        json.dump(data, outfile, indent=4)

if __name__ == "__main__":
    synthetic_data = generate_synthetic_traffic_data(num_samples=1000)
    save_data_to_json(synthetic_data)
    print(f"Synthetic traffic data saved to traffic_data.json")
