import asyncio
import datetime
import json
import math
import os
import random
import re
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

# import logging
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from logging.handlers import TimedRotatingFileHandler

from pydantic import BaseModel
from util import calc_accuracy4math, calc_cloud_score, format_reward_deepseek

logger = logging.getLogger(__name__)


file_handler = TimedRotatingFileHandler(
    "data/reward_log/reward.log",
    when="midnight",
    interval=1,
    backupCount=5,
)
file_handler.setLevel(logging.INFO)  # 设置文件处理器的日志级别
logger.addHandler(file_handler)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BaseRequest(BaseModel):
    data_source: Any
    solution_str: str
    ground_truth: Union[dict, list, str]
    extra_info: Union[dict, list, str]


def format_reward(answer: str | List[str], **kwargs) -> List[float]:
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""

    # step 1
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"

    def drop_end_tag(x: str):
        end_tag_list = ["<|im_end|>", "<|endoftext|>"]

        for temp_end_tag in end_tag_list:
            if x.endswith(temp_end_tag):
                value = x[: -len(temp_end_tag)]
                return value

        value = x
        return value

    if isinstance(answer, list):
        completion_contents = [drop_end_tag(x) for x in answer]
    else:
        completion_contents = [drop_end_tag(answer)]

    matches = [
        re.match(pattern, content, re.DOTALL | re.MULTILINE)
        for content in completion_contents
    ]
    # step1_score = [1.0 if match else 0.0 for match in matches]

    # step 2
    def format_func_patch1(x: str):
        # Check if the string starts with <think> and ends with </answer>
        if not (x[:100].find("<think>") != -1 and x[-100:].find("</answer>") != -1):
            # if not (x[:100].startswith("<think>") and x[-100:].endswith("</answer>")):
            return 0.0

        # Check if each tag appears exactly once
        tags = ["<think>", "</think>", "<answer>", "</answer>"]
        for tag in tags:
            if x.count(tag) != 1:
                return 0.0

        # Check the order of tags
        pos_think_start = x.find("<think>")
        pos_think_end = x.find("</think>")
        pos_answer_start = x.find("<answer>")
        pos_answer_end = x.find("</answer>")

        # Verify correct order
        if not (pos_think_start < pos_think_end < pos_answer_start < pos_answer_end):
            return 0.0

        return 1.0

    final_res = []
    for content_ in completion_contents:
        score = format_func_patch1(content_)
        final_res.append(score)
    # for content_, match in zip(completion_contents, matches):
    #     if match:
    #         score = format_func_patch1(content_)
    #         final_res.append(score)
    #     else:
    #         final_res.append(0.0)

    return final_res


def format_reward_deepseek_wrap(answer: str | List[str], **kwargs) -> List[float]:
    "优化适配deepseek 模型的format reward"
    if isinstance(answer, str):
        return format_reward_deepseek(answer)
    elif isinstance(answer, list):
        return [format_reward_deepseek(x) for x in answer]


def accuracy_reward(pred_answer: List[str], ground_truth: List[str], **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""

    rewards = []
    for pred, truth in zip(pred_answer, ground_truth):
        temp_value = calc_accuracy4math(truth, pred)
        rewards.append(temp_value)

    return rewards


@app.post("/get_reward")
async def get_reward(request: Request):
    json_data = await request.json()

    # print(json_data)

    groud_truth: str = json_data.get("ground_truth", "")
    pred_answer = json_data.get("response_str", "")

    if isinstance(pred_answer, list):
        pred_answer = pred_answer[0]

    # 模拟打分
    score = {
        "format": format_reward([pred_answer])[0],
        "accuracy": accuracy_reward([pred_answer], [groud_truth])[0],
        # "relevance": random.randint(0, 1),
    }
    score["score"] = sum(
        score.values()
    )  # 注意，最后返回的是一个字典。然后这个字典里面一定要有score这个key。值为0 或者1

    cur_date = datetime.datetime.now()

    temp_data = {
        "cur_date": cur_date.strftime("%Y-%m-%d %X"),
        "input_data": json_data,
        "score": score,
    }
    logger.info(json.dumps(temp_data, ensure_ascii=False))

    return score


@app.post("/get_reward2")
async def get_reward2(request: Request):
    """使用辉鹏训练的reward模型，提供反馈接口"""
    json_data = await request.json()

    def wrap_process_data():
        dataset_source = json_data.get("data_source", "")
        pred_answer = json_data.get("response_str", "")

        if dataset_source.find("cloud") == -1:
            groud_truth: str = json_data.get("ground_truth", "")

            if isinstance(pred_answer, list):
                pred_answer = pred_answer[0]

            # 模拟打分
            score = {
                "format": format_reward_deepseek_wrap([pred_answer])[0],
                "accuracy": accuracy_reward([pred_answer], [groud_truth])[0],
                # "relevance": random.randint(0, 1),
            }
            score["score"] = sum(
                score.values()
            )  # 注意，最后返回的是一个字典。然后这个字典里面一定要有score这个key。值为0 或者1

            temp_data = {
                # "cur_date": cur_date.strftime("%Y-%m-%d %X"),
                "input_data": json_data,
                "score": score,
            }

            logger.info(json.dumps(temp_data, ensure_ascii=False))
            return score

        else:
            prompt_str = json_data.get("prompt_str", "")

            reward_detail = calc_cloud_score(prompt_str, pred_answer)

            score = {"score": reward_detail[0]}

            temp_data = {
                # "cur_date": cur_date.strftime("%Y-%m-%d %X"),
                "input_data": json_data,
                "score": score,
                "reason": reward_detail[1],
            }

            logger.info(json.dumps(temp_data, ensure_ascii=False))

            return score

    # Use asyncio to offload potentially CPU-bound task to a thread
    result = await asyncio.to_thread(wrap_process_data)
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=6009)
