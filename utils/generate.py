import json
import random
import time

import openai
from tqdm import tqdm

client = openai.ChatCompletion

themes = {
    "electronic products": [
        "Smartphones",
        "Personal Computers",
        "Televisions",
        "Wearable Devices",
        "Cameras",
        "Audio Systems",
        "Gaming Hardware",
        "Home Automation",
        "Tablets",
        "Drones",
        "Smart Home Devices",
        "E-Readers",
    ],
    "technology": [
        "Artificial Intelligence",
        "Blockchain",
        "Virtual Reality",
        "Cybersecurity Measures",
        "Cloud Solutions",
        "Quantum Devices",
        "Robotics",
        "Network Innovations",
        "Sustainable Energy",
        "Advanced Biotech",
        "Space Exploration Technologies",
    ],
    "fashion": [
        "High Fashion",
        "Urban Streetwear",
        "Designer Brands",
        "Vintage Apparel",
        "Accessories",
        "Children's Wear",
        "SportsWear",
        "Sustainable and Ethical Fashion",
        "Techwear",
        "Seasonal Collections",
    ],
    "food and cuisine": [
        "Plant-based Cuisine",
        "Global Street Eats",
        "Gourmet Dining",
        "Mobile Food Services",
        "Regional Delicacies",
        "Sweets and Confectionery",
        "Health-conscious Foods",
        "International Fusion",
        "Culinary Skills",
        "Beverage Crafting",
    ],
    "movies and entertainment": [
        "Major Studio Releases",
        "Indie Films",
        "Documentaries",
        "Streaming Originals",
        "Celebrity Culture",
        "Awards Season",
        "Reality Shows",
        "Animation",
        "Genre Cinema",
        "Film Festival",
        "Web Series",
        "Fan Culture and Fandom",
    ],
    "health and wellness": [
        "Mental Health Awareness",
        "Fitness Regimens",
        "Dietary Plans",
        "Mindfulness and Meditation",
        "Retreats for Wellbeing",
        "Holistic Medicine",
        "Beauty and Dermatology",
        "Sleep Science, Nutritional Supplements",
        "Wellness Gadgets",
    ],
    "finance and economy": [
        "Equities Market",
        "Savings and Budgeting",
        "Property Market",
        "Pensions and Retirement",
        "Fiscal Policies",
        "Insurance Schemes",
        "Trading Strategies",
        "Financial Tech",
        "International Commerce",
        "Crypto Assets",
    ],
    "sports and athletics": [
        "Team Sports",
        "Basketball",
        "Racquet Sports",
        "Olympic Disciplines",
        "Adventure Sports",
        "Digital Gaming Competitions",
        "Gymnastics",
        "Aquatic Activities",
        "Motorsport",
        "Outdoor Challenges",
        "E-Sports Technology",
        "Urban Sports and Street Games",
    ],
    "travel and tourism": [
        "Offbeat Adventures",
        "Cultural Expeditions",
        "Green Travel",
        "Opulent Journeys",
        "Economical Excursions",
        "Sea Cruises",
        "Solo Explorations",
        "Family Getaways",
        "Heritage Sites",
        "Gastronomic Tours",
    ],
    "art and culture": [
        "Modern Art",
        "Musical Variations",
        "Performing Arts",
        "Literary Works",
        "Exhibition Spaces",
        "Cultural Celebrations",
        "Photographic Arts",
        "Sculptural and Installations",
        "Traditional Crafts",
        "New Media Art",
    ],
}


def load_random_sample(filename):
    with open(filename, "r") as file:
        samples = json.load(file)
        return random.choice(samples)


speaker_count_range = [3, 6]
turns_count_range = [2, 4]

modalities = {
    "T": "The `modality` should always be set to `None`",
    "I": "Use your excellent imagination and strong content generation to add image modalities in the conversation. Please ensure the relevance of the image annotations and use the '<img> content </img>' format to annotate. There can only be a maximum of 1-4 images in the entire conversation. If the utterance has an img annotation, the 'modality' should include 'type', 'caption', 'id'; the 'type' is always 'img', the 'caption' is the corresponding annotation. If there is no image annotation, the 'modality' should be set to `None`",
    "A": "Use your excellent imagination and strong content generation to add audio modalities in the conversation. Please ensure the relevance of the audio annotations and use the '<audio> content </audio>' format to annotate. There can only be a maximum of 1-4 audios in the entire conversation. If the utterance has an audio annotation, the 'modality' should include 'type', 'caption', 'id'; the 'type' is always 'audio', the 'caption' is the corresponding annotation. If there is no audio annotation, the 'modality' should be set to `None`",
    "V": "Use your excellent imagination and strong content generation to add video modalities in the conversation. Please ensure the relevance of the video annotations and use the '<video> content </video>' format to annotate. There can only be a maximum of 1-4 videos in the entire conversation. If the utterance has a video annotation, the 'modality' should include 'type', 'caption', 'id'; the 'type' is always 'video', the 'caption' is the corresponding annotation. If there is no video annotation, the 'modality' should be set to `None`",
    "IA": "Use your excellent imagination and strong content generation to add image and audio modalities in the conversation. Please ensure the relevance of the image and audio annotations and use the '<img> content </img>', '<audio> content </audio>', format to annotate. There can only be a maximum of 2-5 images and audios in the entire conversation. If the utterance has an image or audio annotation, the 'modality' should include 'type', 'caption', 'id'; the 'type' is always 'img' or 'audio', the 'caption' is the corresponding annotation. If there is no image or audio annotation, the 'modality' should be set to `None`",
    "IV": "Use your excellent imagination and strong content generation to add image and video modalities in the conversation. Please ensure the relevance of the image and video annotations and use the '<img> content </img>', '<video> content </video>' format to annotate. There can only be a maximum of 2-5 images and videos in the entire conversation. If the utterance has an image or video annotation, the 'modality' should include 'type', 'caption', 'id'; the 'type' is always 'img' or 'video', the 'caption' is the corresponding annotation. If there is no image or video annotation, the 'modality' should be set to `None`",
    "AV": "Use your excellent imagination and strong content generation to add audio and video modalities in the conversation. Please ensure the relevance of the audio and video annotations and use the '<audio> content </audio>', '<video> content </video>' format to annotate. There can only be a maximum of 2-5 audios and videos in the entire conversation. If the utterance has an audio or audio annotation, the 'modality' should include 'type', 'caption', 'id'; the 'type' is always 'audio' or 'video', the 'caption' is the corresponding annotation. If there is no audio or video annotation, the 'modality' should be set to`None`",
    "IAV": "Use your excellent imagination and strong content generation to add image, audio and video modalities in the conversation. Please ensure the relevance of the image, audio and video annotations and use the '<img> content </img>', '<audio> content </audio>', '<video> content </video>' format to annotate. There can only be a maximum of 3-6 images, audios and videos in the entire conversation. If the utterance has an image, audio or video annotation, the 'modality' should include 'type', 'caption', 'id'; the 'type' is always 'img', 'audio' or 'video', the 'caption' is the corresponding annotation. If there is no image, audio or video annotation, the 'modality' should be set to `None`",
}

modality_files = {
    "T": "xxx.json",
    "I": "xxx.json",
    "A": "xxx.json",
    "V": "xxx.json",
    "IA": "xxx.json",
    "IV": "xxx.json",
    "AV": "xxx.json",
    "IAV": "xxx.json",
}


def generate_dialogue(retry_limit=5, sleep_time=10):
    speaker_count = random.randint(*speaker_count_range)
    turn_count = random.randint(*turns_count_range)

    main_theme = random.choice(list(themes.keys()))

    subtheme = random.choice(themes[main_theme])
    selected_modality_key = "I"
    selected_modality_description = modalities[selected_modality_key]

    if selected_modality_key in modality_files:
        sample_json = load_random_sample(modality_files[selected_modality_key])
        sample_json_string = json.dumps(sample_json, indent=4)

    content_part1 = (
        f"```\n"
        f"Please comply with the following instructions. Do not comment, judge, or output other texts and only return the results.\n"
        f"1. Generate a nonlinear dialogue replying structure between / among {speaker_count} speakers, and the turns of the dialogue must be {turn_count}.\n"
        f"2. Each speaker in the dialogue should have a unique `speaker_id` and a unique `speaker_name`, and each dialogue should have a unique `doc_id`.\n"
        f"3. Dialogue should incorporate discussions around one or more `targets' which are the main objects being discussed, such as electronic devices like smartphones or cameras. These discussions should focus on specific `aspects' of the `targets', such as 'screen' or 'battery' for smartphones, attributed to a `holder'. Each `aspect' is evaluated through `opinions', such as 'good', 'bad', or 'a drawback', which are supported by `rationales' explaining the reasoning behind these opinions. All discussions must align with real-world conversational context around these devices.\n"
    )

    content_part2 = (
        "4. Annotate and `order` the occurrence of `holder', `target', `aspect', `opinion', and `rationale' in HTML format in the `annotation'. In utterances that incorporate multimodal elements, one of these elements must be `implicit`, its value appearing exclusively in the multimodal caption and not mentioned in the dialogue text. All other elements must be explicitly mentioned in the dialogue text and marked as `explicit`.\n"
        "5. Every utterance except the first utterance is a `reply` to dialogue sentence with index n, the reply property of this utterance should be n, the first utterance is -1.\n"
        "6. The conversation must include all five elements: `holder', `target', `aspect', `opinion', and `rationale'. In utterances that include multimodal elements, designate one element as `implicit` by including its value only in the multimodal caption. This value should not appear in the dialogue text, while all other elements must be mentioned explicitly in the text.\n"
        "7. Store all parts of the conversation, including any multimedia modalities, in accordance with the provided example format. For each multimodal utterance, include the `implicit` element's value only in the caption, detailing the `type`, `caption`, and `id` of the modality. Decide a sentiment for each combination based on the corresponding opinion, with sentiments being either positive, negative, or neutral.\n"
        f"8. {selected_modality_description}\n\n"
        f"9. Ensure full comprehension of the provided example and apply it to create a dialogue that meets all specified criteria, including the proper integration of multimodal elements. Strictly adhere to the example `json` format for organizing the storage structure of the generated dialogue, as shown in the provided example.\n"
        f"For instance, a sample `json` output would be:\n"
        f"{sample_json_string}\n"
        "```"
    )

    content_string = content_part1 + content_part2
    # print(content_string)

    for attempt in range(retry_limit):
        try:
            response = client.create(
                model="gpt-4-turbo-preview",
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": f"As an expert playwright skilled in crafting dialogues, you're tasked with generating conversations centered around {subtheme}.",
                    },
                    {"role": "user", "content": content_string},
                ],
            )
            return json.loads(response.choices[0].message["content"].strip())
        except openai.error.Timeout as e:
            if attempt < retry_limit - 1:
                time.sleep(sleep_time)
                continue
            else:
                raise
        except Exception as e:
            raise


def main():
    openai.api_key = "xxxxxxxxxxxxx"
    output_file = "./xxx.json"

    try:
        with open(output_file, "r") as file:
            dialogues = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        dialogues = []

    for _ in tqdm(range(100), desc="Generating dialogues"):
        dialogue = generate_dialogue()
        dialogues.append(dialogue)

        with open(output_file, "w") as file:
            json.dump(dialogues, file, indent=4)


if __name__ == "__main__":
    main()
