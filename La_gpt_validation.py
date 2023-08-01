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
    model_path = (
        "/var/task/klue_finetuned_distilled_ckpt/3-klue_distilled/checkpoint-3856"
    )
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
    total = 0
    for d in dc:
        total += dc[d]
        print(f"label: {d}, count: {dc[d]}\n")

    # gpt글 reject 기준
    check = 0
    for label in range(3, 7):
        if label in dc:
            check += dc[label]

    if check / total > 0.5:
        print("reject")
        # sqs 재전송 (SQS_make_story)
        try:
            # 최종 처리는 sqs에 연결된 lambda가 진행
            sqs = boto3.resource("sqs", region_name="ap-northeast-2")
            queue = sqs.get_queue_by_name(QueueName=f"SQS_make_story_{env}")

            temp_json = {}
            temp_json["user_id"] = user_id
            temp_json["time_stamp"] = time_stamp
            message_body = json.dumps(temp_json)
            response = queue.send_message(
                MessageBody=message_body,
            )
        except:
            print("sqs fail!")
    else:
        try:
            # to SQS_post_midjourney_story
            sqs = boto3.resource("sqs", region_name="ap-northeast-2")
            queue = sqs.get_queue_by_name(QueueName=f"SQS_post_midjourney_story_{env}")
            temp_json = {}
            temp_json["user_id"] = user_id
            temp_json["time_stamp"] = time_stamp
            message_body = json.dumps(temp_json)
            response = queue.send_message(
                MessageBody=message_body,
            )
        except:
            print("sqs fail!")

    print("good")
    return {"statusCode": 200, "body": json.dumps("Hello from Lambda!")}
