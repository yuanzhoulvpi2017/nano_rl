import re
from typing import List
from math_verify import LatexExtractionConfig, StringExtractionConfig, parse, verify
import requests


def extract_answer(text):
    """ """
    pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) == 1:
        return matches[0]
    else:
        return None


def extract_answer_deepseek(text):
    """ """
    pattern = r"</think>\n(.*?)<｜end▁of▁sentence｜>"
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) == 1:
        return matches[0]
    else:
        return None


def extract_user_query_deepseek(text):
    """ """
    pattern = r"<｜User｜>(.*?)<｜Assistant｜><think>\n"
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) == 1:
        return matches[0]
    else:
        return None


def extract_boxed(text):
    """使用计数方法提取\boxed{}中的内容，处理任意嵌套的大括号"""
    if not isinstance(text, str):
        return None

    boxed_start = text.find(r"\boxed{")
    if boxed_start == -1:
        return None

    # 跳过'\boxed{'
    start_pos = boxed_start + 7
    brace_count = 1
    end_pos = start_pos

    # 通过计数大括号来找到匹配的结束大括号
    while end_pos < len(text) and brace_count > 0:
        if text[end_pos] == "{":
            brace_count += 1
        elif text[end_pos] == "}":
            brace_count -= 1
        end_pos += 1

    if brace_count == 0:
        # 减去1是因为我们要排除最后的大括号
        return text[start_pos : end_pos - 1]

    return None


def sub_answer_by_math(x: List[str] | str | None):
    if x is None:
        return []
    if isinstance(x, str):
        x = [x]

    res = [] + x

    select_config = [
        StringExtractionConfig(),
        LatexExtractionConfig(),
        # LatexExtractionConfig(),
    ]

    for temp_x in x:
        res.extend(parse(temp_x, extraction_mode="first_match"))
        for temp_config in select_config:
            res.extend(
                parse(
                    temp_x,
                    extraction_mode="first_match",
                    extraction_config=[temp_config],
                )
            )

    res = list(set(res))
    return res


def get_model_gen_result(x: str):
    # 用于提取模型输出的最终结果
    if not isinstance(x, str):
        return []
    else:
        answer_part = extract_answer_deepseek(x)
        if answer_part is None:
            return []
        else:
            box_part = extract_boxed(answer_part)
            box_part = [] if box_part is None else box_part

            math_value = sub_answer_by_math(box_part)
            # math_value.extend()

            math_value = list(set(math_value))
            return math_value


def get_ground_truth(x: str):
    # 处理ground truth的结果
    res = sub_answer_by_math(x)
    res = list(set(res))
    return res


def calc_accuracy4math(ground_truth: str, response: str) -> float:
    """计算数学公式的准确率"""
    # 提取模型输出的最终结果
    model_gen_result = get_model_gen_result(response)
    # 提取标准答案的最终结果
    ground_truth_result = get_ground_truth(ground_truth)

    # 计算准确率
    accuracy = verify(model_gen_result, ground_truth_result)
    accuracy = accuracy * 1.0
    return accuracy


def get_cloud_score_api(query: str, response: str):
    try:
        api_url = "http://10.xxx.0.xxx:7009/pooling"

        # Input like Chat API
        prompt = {
            "model": "r1-reward",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a helpful assistant."}
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": query}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": response}],
                },
            ],
        }
        headers = {"Content-Type": "application/json", "Authorization": "Bearer EMPTY"}
        response = requests.post(api_url, headers=headers, json=prompt)

        # score = {"score": response.json()["data"][0]["data"][0]}
        score = response.json()["data"][0]["data"][0]
        score = float(score)

        return score
    except Exception as e:
        return 0.0


def get_reward_score_api_cloud(user_prompt: str, response_str):
    response = requests.post(
        "http://10.xxx.240.xxx:5008/reward/api",
        json={"user_prompt": user_prompt, "response": response_str},
    )
    return response.json()


def calc_cloud_score(query: str, response: str) -> float:
    """使用自定义的reward接口进行训练"""
    model_response_answer = extract_answer_deepseek(response)
    query = extract_user_query_deepseek(query)

    if query is None:
        return -99.0, None
    if model_response_answer is None:
        return -99.0, None
    else:
        score = get_reward_score_api_cloud(query, model_response_answer)
        return float(score.get("reward_score")), str(score.get("critique"))
