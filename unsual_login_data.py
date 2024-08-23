import openai
import json

# Set your OpenAI API key
openai.api_key = "LA-0bb4044544834ffcb870053aebf7a750a7e616f60e20471596fd02545d258041"

def generate_synthetic_data_via_api(num_samples=1000):
    synthetic_data = []

    for _ in range(num_samples):
        # Use a descriptive prompt to generate data
        prompt = (
            "Generate a synthetic login event with the following details:\n"
            "1. Timestamp (hour between 0 and 23)\n"
            "2. Location (US, Europe, Asia, Africa, Australia)\n"
            "3. Device (Desktop, Mobile, Tablet)\n"
            "4. Is the login unusual? (yes/no)"
        )

        response = openai.Completion.create(
            engine="text-davinci-003",  # Use a relevant GPT model
            prompt=prompt,
            max_tokens=60,
            temperature=0.7
        )

        # Parse the response to extract the generated data
        response_text = response.choices[0].text.strip()

        # Split the generated text into parts
        parts = response_text.split("\n")
        try:
            timestamp = int(parts[0].split(":")[1].strip())
            location = parts[1].split(":")[1].strip()
            device = parts[2].split(":")[1].strip()
            is_unusual = parts[3].split(":")[1].strip()

            # Map 'yes'/'no' to 1/0 for labels
            label = 1 if is_unusual.lower() == "yes" else 0

            synthetic_data.append({
                'timestamp': timestamp,
                'location': location,
                'device': device,
                'label': label
            })
        except:
            print(f"Error parsing data: {response_text}")

    # Save generated data as JSON
    with open('synthetic_login_data_api.json', 'w') as f:
        json.dump(synthetic_data, f, indent=4)

    print(f"{num_samples} samples generated and saved to synthetic_login_data_api.json")

if __name__ == "__main__":
    generate_synthetic_data_via_api(num_samples=100)
