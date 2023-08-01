import json
import boto3
import torch
import os
import re
from transformers import AutoTokenizer, BertForSequenceClassification


def predict_labels(inputs, model_path, labels):
    tokenizer = AutoTokenizer.from_pretrained(model_path)  # Load the tokenizer
    model = BertForSequenceClassification.from_pretrained(
        model_path
    )  # Load the fine-tuned BERT model (it will be on CPU by default)
    model.eval()  # Ensure the model is in evaluation mode

    inputs_encoded = tokenizer(
        inputs, padding=True, truncation=True, return_tensors="pt"
    )  # Tokenize the input texts and convert them to tensors

    with torch.no_grad():  # Make predictions
        outputs = model(**inputs_encoded)

    # Get the predicted labels
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_labels = [labels[i] for i in torch.argmax(probabilities, dim=1)]

    return predicted_labels


"""
Part for inferernce
"""

def lambda_handler(event, context):
    print(event)
    # version check
    function_arn = context.invoked_function_arn
    env = function_arn.split(":")[-1]
    print(env)
    env = "prod"

    # parsing message
    try:
        message_body = json.loads(event["Records"][0]["body"])
        user_id = message_body["user_id"]
        time_stamp = message_body["time_stamp"]
    except:
        print("parsing fail!")

    # get story
    try:
        dynamodb_client = boto3.client("dynamodb")
        query = f"SELECT * FROM Dy_gpt_story_{env} where user_id='{user_id}' and time_stamp='{time_stamp}'"
        print(query)
        result_story = dynamodb_client.execute_statement(Statement=query)
    except:
        print("story fail!")

    # get story keys
    try:
        temp = list(result_story["Items"][0].keys())
        temp.remove("user_id")
        temp.remove("time_stamp")
        temp.sort()
        story_keys = temp
    except:
        print("parsing fail!")
        return {"statusCode": 200, "body": json.dumps("Hello")}

    # page, sentence parsing -> . [SEP]
    inputs = []
    for page_num in range(len(story_keys)):
        text = result_story["Items"][0][story_keys[page_num]]["S"]
        sentence = re.split("(?<=[.!?]) +", text)
        inputs.append(" [SEP]".join(sentence))

    print(inputs)

    labels = [0, 1, 2, 3, 4, 5, 6]  # Replaced with our own label names
    model_path = "./test_trainer_best_model_0728_teache student, data follows distribution/checkpoint-3374"  # checkpoint path for fine-tuned model
    ar = []  # to store predictd labels
    dc = {}  # to count predicted labels
    predicted_labels = predict_labels(
        inputs, model_path, labels
    )  # send to model for prediction

    for predicted_label in predicted_labels:
        ar.append(int(predicted_label))

    # part for sorting and counting numbers of predicted label
    for value in ar:
        if value in dc:
            dc[value] += 1
        else:
            dc[value] = 1
    for d in dc:
        print(f"label: {d}, count: {dc[d]}\n")
        
    # gpt글 reject 기준
    
    
