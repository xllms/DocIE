import os
import sys

sys.path.append("./")
import json

import re
from tqdm import tqdm

from typing import List, Optional
from pydantic import BaseModel, Field, create_model

import argparse
from openai import AzureOpenAI


REGION = ""
MODEL = "gpt-4o-2024-08-06"
API_KEY = ""
API_BASE = ""
ENDPOINT = f"{API_BASE}/{REGION}"
client = AzureOpenAI(
    api_key=API_KEY,
    api_version="2024-08-01-preview",
    azure_endpoint=ENDPOINT,
)


def getFormat(schema_list, task):
    if task == "NER":
        return NERFormat(schema_list)
    elif task == "RE":
        return REFormat(schema_list)


def NERFormat(schema):
    tmp = {}
    for field in schema:
        tmp[field] = (Optional[List[str]], Field(default_factory=list))
    NEROutputModel = create_model("NEROutputModel", **tmp)
    return NEROutputModel


def REFormat(schema):
    class RelationItem(BaseModel):
        subject: str
        object: str

    REOutputModel = create_model(
        "REOutputModel",
        **{
            relation: (Optional[List[RelationItem]], Field(default_factory=list))
            for relation in schema
        },
    )
    return REOutputModel


def getResponse(prompt, text, schema_list, task):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": f"{prompt}"},
            {"role": "user", "content": f"{text}"},
        ],
        response_format=getFormat(schema_list, task),
    )

    message = completion.choices[0].message
    if message.parsed:
        # print(message.parsed.json())
        return message.parsed.json()
    else:
        print("=============== GPT-4o Parses Failed! ===========")
        return []


def inference(args):
    records = []
    with open(args.input_path, "r") as reader:
        for line in reader:
            data = json.loads(line)
            data["output"] = "test"
            records.append(data)

    with open(args.output_path, "w") as writer:
        for data in tqdm(records):
            data_dict = eval(data["instruction"])
            prompt = data_dict["instruction"]
            text = data_dict["input"]
            schema_list = data_dict["schema"]
            result = getResponse(prompt, text, schema_list, args.task)
            print(result)
            data["output"] = result
            writer.write(json.dumps(data, ensure_ascii=False) + "\n")


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Please give the argument that GPT-4o need!"
    )
    parser.add_argument("-i", "--input_path", type=str, help="")
    parser.add_argument("-o", "--output_path", type=str, help="")
    parser.add_argument("-t", "--task", type=str, choices=["NER", "RE"], help="NER/RE")
    args = parser.parse_args()
    inference(args)


if __name__ == "__main__":
    main()
