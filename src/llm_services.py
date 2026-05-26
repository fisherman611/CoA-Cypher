import re
import json
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from src.logger_config import setup_logger

logger = setup_logger(__name__)


def init_llm_chain(configs, prompt_template):
    prompt = PromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(**configs)
    chain = prompt | llm | StrOutputParser()
    return chain


def init_llm(configs):
    return ChatOpenAI(**configs)


def call_llm(llm, prompt_template, input):
    system_prompt = prompt_template["system"]
    user_prompt = prompt_template["user"]
    messages = [
        {"role": "system", "content": system_prompt.format(**input)},
        {"role": "user", "content": user_prompt.format(**input)},
    ]
    response = llm.invoke(messages).content
    parsed_response = parse_llm_response(response)
    return parsed_response


def parse_json_from_string(text):
    try:
        # 1. Tìm block JSON đầu tiên
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("Không tìm thấy block JSON trong chuỗi.")
        json_block = match.group(0)

        # 2. Chuẩn hoá boolean kiểu Python
        json_block = (
            json_block.replace("True", "true")
            .replace("False", "false")
            .replace("None", "null")
        )

        # 3.txt. Xử lý triple quotes: """...""" → "..."
        def convert_triple_quotes(match):
            content = match.group(1)
            content = content.replace("\n", "\\n").replace(
                '"', '\\"'
            )  # escape newline và dấu "
            return f'"{content}"'

        json_block = re.sub(
            r'"""(.*?)"""', convert_triple_quotes, json_block, flags=re.DOTALL
        )

        # 4. Escape \n trong chuỗi "..." thường
        def escape_newlines_in_string(m):
            return m.group(0).replace("\n", "\\n")

        json_block = re.sub(
            r'"(.*?)"', escape_newlines_in_string, json_block, flags=re.DOTALL
        )

        # 5. Parse JSON
        return json.loads(json_block)

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse JSON: {e}")
        logger.error(f"Original text: {text}")
        return None
    except Exception as e:
        logger.error(f"Unknown error while parsing JSON: {e}")
        logger.error(f"Original text: {text}")
        return None


def parse_llm_response(response: str):
    """
    Tách phần <think>...</think> và phần câu trả lời cuối cùng.

    Args:
        response (str): Output từ LLM có chứa reasoning.

    Returns:
        dict: {
            "think": str,          # nội dung trong <think>
            "final_answer": str    # phần còn lại sau khi bỏ reasoning
        }
    """
    if not response or not isinstance(response, str):
        return {"think": "", "final_answer": ""}

    # Tìm nội dung trong cặp <think>...</think>
    match = re.search(
        r"<think\s*>\s*(.*?)\s*</think\s*>", response, flags=re.IGNORECASE | re.DOTALL
    )

    if match:
        think = match.group(1).strip()
        # Loại bỏ toàn bộ đoạn <think>...</think> để lấy phần trả lời cuối
        final_answer = re.sub(
            r"<think\s*>.*?</think\s*>", "", response, flags=re.IGNORECASE | re.DOTALL
        ).strip()
    else:
        think = ""
        final_answer = response.strip()

    return {"think": think, "final_answer": final_answer}
