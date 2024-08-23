import openai
import json

# Initialize OpenAI client with your API key
openai.api_key = "LA-0bb4044544834ffcb870053aebf7a750a7e616f60e20471596fd02545d258041"

def generate_synthetic_data_via_api(num_samples=100):
    synthetic_data = []

    for _ in range(num_samples):
        # Use a descriptive prompt to generate data
        prompt = (
            "Generate a synthetic login event with the following details in this format:\n"
            "1. Timestamp: <hour between 0 and 23>\n"
            "2. Location: <US, Europe, Asia, Africa, Australia>\n"
            "3. Device: <Desktop, Mobile, Tablet>\n"
            "4. Is the login unusual?: <yes or no>"
        )

        try:
            response = openai.Completion.create(
                engine="text-davinci-003",  # Use a relevant GPT model
                prompt=prompt,
                max_tokens=60,
                temperature=0.7
            )

            # Parse the response to extract the generated data
            response_text = response.choices[0].text.strip()

            # Split the generated text into lines and extract values
            lines = response_text.split("\n")
            if len(lines) < 4:
                print(f"Incomplete response, skipping: {response_text}")
                continue

            timestamp = int(lines[0].split(":")[1].strip())
            location = lines[1].split(":")[1].strip()
            device = lines[2].split(":")[1].strip()
            is_unusual = lines[3].split(":")[1].strip()

            # Map 'yes'/'no' to 1/0 for labels
            label = 1 if is_unusual.lower() == "yes" else 0

            synthetic_data.append({
                'timestamp': timestamp,
                'location': location,
                'device': device,
                'label': label
            })

        except Exception as e:
            print(f"Error processing response: {str(e)}")

        # Save generated data as JSON
    with open('synthetic_login_data_api.json', 'w') as f:
        json.dump(synthetic_data, f, indent=4)

    print(f"{num_samples} samples generated and saved to synthetic_login_data_api.json")

if __name__ == "__main__":
    generate_synthetic_data_via_api(num_samples=100)
