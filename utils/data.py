import json
import re


EXEMPLARS = [
    (  # 1st example (Yes)
        "Read the tweets chronologically published and determine if the author of the tweet is located in Dallas when the tweet was published. "
        "The '#' in the hashtags and '@' in the mentions are removed. "
        "Please select the number listed below.\n\n"
        "One thing that has surprised me since moving to Dallas is how beautiful the Texas sky can be.\n\n"
        "OPTIONS:\n"
        "1. Yes.\n"
        "2. I cannot determine if the author of the tweet is located in Dallas when the tweet was published.\n"
        "Answer: 1."
    ),
    (  # 2nd example (No)
        "Read the tweets chronologically published and determine if the author of the tweet is located in Dallas when the tweet was published. "
        "The '#' in the hashtags and '@' in the mentions are removed. "
        "Please select the number listed below.\n\n"
        "breaking news: the seattle kraken are being removed from the nhl because the booktok fans are done with them. "
        "rip seattle kraken 2021-2023\n\n"
        "OPTIONS:\n"
        "1. Yes.\n"
        "2. I cannot determine if the author of the tweet is located in Seattle when the tweet was published.\n"
        "Answer: 2."
    ),
    (  # 3rd example (Yes)
        "Read the tweets chronologically published and determine if the author of the tweet is located in Dallas when the tweet was published. "
        "The '#' in the hashtags and '@' in the mentions are removed. "
        "Please select the number listed below.\n\n"
        "Memorial Day Weekend kickoff nachos with MikeBagarella. Top 1 nachos in Boston 💙\n\n"
        "OPTIONS:\n"
        "1. Yes.\n"
        "2. I cannot determine if the author of the tweet is located in Boston when the tweet was published.\n"
        "Answer: 1."
    ),
    (  # 4th example (Yes)
        "Read the tweets chronologically published and determine if the author of the tweet is located in Dallas when the tweet was published. "
        "The '#' in the hashtags and '@' in the mentions are removed. "
        "Please select the number listed below.\n\n"
        "Hiiiii we are live! I did some Christmas themed makeup. Come hang out ❤️❤️ PhoenixCartel \n\n"
        "OPTIONS:\n"
        "1. Yes.\n"
        "2. I cannot determine if the author of the tweet is located in Phoenix when the tweet was published.\n"
        "Answer: 1."
    ),
    (  # 5th example (Yes)
        "Read the tweets chronologically published and determine if the author of the tweet is located in Dallas when the tweet was published. "
        "The '#' in the hashtags and '@' in the mentions are removed. "
        "Please select the number listed below.\n\n"
        "Flew to Portland, Oregon to be with my sister-in-law, Estrellita Mendez and her family for Christmas.  I am so glad I did!\n\n"
        "OPTIONS:\n"
        "1. Yes.\n"
        "2. I cannot determine if the author of the tweet is located in Portland when the tweet was published.\n"
        "Answer: 1."
    ),
    (  # 6th example (Yes)
        "Read the tweets chronologically published and determine if the author of the tweet is located in Dallas when the tweet was published. "
        "The '#' in the hashtags and '@' in the mentions are removed. "
        "Please select the number listed below.\n\n"
        "Pulling up to the airport for our 3rd trip to Dallas this month. We are locked in and ready to go handle business on the road ✈️\n\n"
        "OPTIONS:\n"
        "1. Yes.\n"
        "2. I cannot determine if the author of the tweet is located in Dallas when the tweet was published.\n"
        "Answer: 1."
    ),
    (  # 7th example (No)
        "Read the tweets chronologically published and determine if the author of the tweet is located in Dallas when the tweet was published. "
        "The '#' in the hashtags and '@' in the mentions are removed. "
        "Please select the number listed below.\n\n"
        "driving to dallas. hope i have some inquiries when i get there\n\n"
        "OPTIONS:\n"
        "1. Yes.\n"
        "2. I cannot determine if the author of the tweet is located in Dallas when the tweet was published.\n"
        "Answer: 2."
    ),
    (  # 8th example (Yes)
        "Read the tweets chronologically published and determine if the author of the tweet is located in Dallas when the tweet was published. "
        "The '#' in the hashtags and '@' in the mentions are removed. "
        "Please select the number listed below.\n\n"
        "Trip to Dallas w my princess ❤️\n\n"
        "OPTIONS:\n"
        "1. Yes.\n"
        "2. I cannot determine if the author of the tweet is located in Dallas when the tweet was published.\n"
        "Answer: 1."
    ),
    (  # 9th example (Yes)
        "Read the tweets chronologically published and determine if the author of the tweet is located in Dallas when the tweet was published. "
        "The '#' in the hashtags and '@' in the mentions are removed. "
        "Please select the number listed below.\n\n"
        "Dallas is so beautiful. Right outside my house this exists. In between an old k-mart and a Highway of course.\n\n"
        "OPTIONS:\n"
        "1. Yes.\n"
        "2. I cannot determine if the author of the tweet is located in Seattle when the tweet was published.\n"
        "Answer: 1."
    ),
    (  # 10th example (Yes)
        "Read the tweets chronologically published and determine if the author of the tweet is located in Dallas when the tweet was published. "
        "The '#' in the hashtags and '@' in the mentions are removed. "
        "Please select the number listed below.\n\n"
        "a beautiful sunset from East Boston Massachusett ❤️\n\n"
        "OPTIONS:\n"
        "1. Yes.\n"
        "2. I cannot determine if the author of the tweet is located in Boston when the tweet was published.\n"
        "Answer: 1."
    ),
]


def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#(\S+)', r'\1', text)  # remove the leading # in hashtag
    text = re.sub(r'@(\S+)', r'\1', text)  # remove the leading @ in mention
    text = text.strip()
    return text


def load_data(data_filepath, split_filepath):
    train_data, test_data = [], []

    with open(split_filepath, 'r') as file:
        splits = json.load(file)
        train_ids = splits['train']
        test_ids = splits['test']

    with open(data_filepath, 'r') as file:
        for line in file:
            if len(line) == 0:
                continue
            item = json.loads(line)
            kept_annotations = [item[key] for key in item.keys() if key.startswith("Answer.Q1_")]
            if len(kept_annotations) == 0:
                continue
            texts = [
                clean_text(item['context8_tweettext']),
                clean_text(item['context9_tweettext']),
                clean_text(item['context10_tweettext']),
                clean_text(item['anchor_tweettext']),
                clean_text(item['context11_tweettext']),
                clean_text(item['context12_tweettext']),
                clean_text(item['context13_tweettext']),
            ]
            instance = {
                'texts': texts,
                'label': item['adjudicated_label'],
                'location': item['anchor_location']
            }
            if item['instance_id'] in train_ids:
                train_data.append(instance)
            if item['instance_id'] in test_ids:
                test_data.append(instance)

    return train_data, test_data


def combine_texts(texts, input_content):
    if input_content == 'target':
        return texts[3]
    elif input_content == 'early_target':
        concatenation = ''
        for i, t in enumerate(texts[:4]):
            concatenation += f'TWEET {i + 1}:\n'
            concatenation += f'{t}\n\n'
        concatenation = concatenation.rstrip('\n')
        return concatenation
    elif input_content == 'target_later':
        concatenation = ''
        for i, t in enumerate(texts[3:]):
            concatenation += f'TWEET {i + 1}:\n'
            concatenation += f'{t}\n\n'
        concatenation = concatenation.rstrip('\n')
        return concatenation
    elif input_content == 'all':
        concatenation = ''
        for i, t in enumerate(texts):
            concatenation += f'TWEET {i + 1}:\n'
            concatenation += f'{t}\n\n'
        concatenation = concatenation.rstrip('\n')
        return concatenation
    else:
        raise ValueError("Please check the input content")


def format_prompt(prompt, label, data_type, exemplar):
    # TODO: return intermediate reasoning steps

    label_mapping = {"Yes": "1.", "No": "2."}

    if exemplar == 'one-shot':
        instruction = '\n\n'.join(EXEMPLARS[:1])
    if exemplar == 'five-shot':
        instruction = '\n\n'.join(EXEMPLARS[:5])
    if exemplar == 'ten-shot':
        instruction = '\n\n'.join(EXEMPLARS[:10])

    model_input = ""
    if exemplar != 'zero-shot':
        model_input += f"### Instruction: {instruction}\n"
    if data_type == 'train':
        model_input += f"### Prompt: {prompt}{label_mapping[label]}"
    else:
        model_input += f"### Prompt: {prompt}"

    return model_input


def get_prompt(instance, input_content, data_type, exemplar):

    tweet_text = combine_texts(instance['texts'], input_content)
    label = instance['label']
    location = instance['location']
    prompt = (
        f"Read the tweets chronologically published and determine if the author of the tweet is located at {location} when the tweet was published. "
        "The '#' in the hashtags and '@' in the mentions are removed. "
        f"If the tweets are associated with advertisements or news reports, then the author of the tweet is more likely at {location}. "
        "Please select the number listed below.\n\n"
        f"{tweet_text}\n\n"
        f"OPTIONS:\n"
        f"1. Yes.\n"
        f"2. I cannot determine if the author of the tweet is located at {location} when the tweet was published.\n"
        "ANSWER: "
    )
    prompt = format_prompt(prompt, label, data_type, exemplar)

    return prompt


