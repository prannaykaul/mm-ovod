import argparse
import time
import json

import openai
from openai.error import RateLimitError
from lvis import LVIS


API_KEY = "YOUR_API_KEY"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ann-path",
        type=str,
        default="datasets/lvis/lvis_v1_val.json"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="datasets/metadata/lvis_gpt3_{}_descriptions_own.json"
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="text-davinci-002"
    )


def main(args):
    lvis_gt = LVIS(args.ann_path)
    categories = sorted(lvis_gt.cats.values(), key=lambda x: x["id"])

    category_list = [c['synonyms'][0].replace('_', ' ') for c in categories]
    all_responses = {}
    vowel_list = ['a', 'e', 'i', 'o', 'u']

    for i, category in enumerate(category_list):
        if category[0] in vowel_list:
            article = 'an'
        else:
            article = 'a'
        prompts = []
        prompts.append("Describe what " + article + " " + category + " looks like.")

        all_result = []
        # call openai api taking into account rate limits
        for curr_prompt in prompts:
            try:
                response = openai.Completion.create(
                    model="text-davinci-002",
                    prompt=curr_prompt,
                    temperature=0.99,
                    max_tokens=50,
                    n=10,
                    stop="."
                )
            except RateLimitError:
                print("Hit rate limit. Waiting 15 seconds.")
                time.sleep(15)
                response = openai.Completion.create(
                    model="text-davinci-002",
                    prompt=curr_prompt,
                    temperature=.99,
                    max_tokens=50,
                    n=10,
                    stop="."
                )

            time.sleep(0.15)

            for r in range(len(response["choices"])):
                result = response["choices"][r]["text"]
                all_result.append(result.replace("\n\n", "") + ".")
        all_responses[category] = all_result

    output_path = args.output_path.format(args.openai_model)
    with open(output_path, 'w') as f:
        json.dump(all_responses, f, indent=4)


if __name__ == "__main__":
    args = get_args()
    main(args)
