from openai import OpenAI

class LLM:
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="sk-947ed4a81cb643d685471ff531c4ec59",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    @classmethod
    def get_flight_info(cls, date, belonging_flight, user_info, dialogue):
        content = f'你是一个航班座位选择助手，能够根据用户的输入和提供的数据，帮助用户选择合适的航班或座位。\n\
以下是你的任务说明和规则：\n\
当前的日期是 {date}。\n\
用户名下航班信息 JSON 如下：\n{belonging_flight}\n\
id 表示航班唯一标识符，class 表示所属航班的舱位类型， first 表示头等舱， business 表示商务舱， economy 表示经济舱，\n\
用户信息 JSON 如下：\n{user_info}\n\
flight 表示用户当前是否已指定航班，"None" 表示未指定。\n\
根据用户输入的需求，返回以下格式的结果：\n\
[航班ID, 座位类型]\n\
航班ID ： 如果用户在对话中提到了航班，则返回对应要求的航班 ID；如果用户未提到航班，则这一部分为 None。\n\
座位类型 ： 如果用户在对话中提到了座位，则返回对应要求的座位类型；\n\
其中，如果用户提到 “靠窗” 或者类似的意思，则这一部分为 "window-random" ；\n\
如果用户提到 “靠走廊” 、 “靠过道” 或者类似的意思，则这一部分为 "aisle-random" ；\n\
如果用户提到了具体的X排Y座的座位，则这一部分为 "X-Y" ；\n\
如果用户未提到座位，则这一部分为 None。\n\
除此之外，这一部分为 None。\n\
你只需要输出结果数组，不需要输出任何提示信息。'
        completion = cls.client.chat.completions.create(
            model="qwen-max",
            messages=[
                {
                    'role': 'system',
                    'content': content
                },
                {
                    'role': 'user',
                    'content': dialogue
                }
            ],
        )

        return completion.choices[0].message.content

    @classmethod
    def check_if_gibberish(cls, text) -> bool | None:
        content = f'你是一个语言识别助手，你要根据文字的输入，判断这一句是否有大量重复的内容，例如重复了很多次的"I\'ll be going“等等。\n以下是需要判断的文字：\n{text}\n请你输出一个数字，数字为0表示正常，数字为1表示不正常。你只需要输出一个数字，不需要输出任何提示信息。'
        completion = cls.client.chat.completions.create(
            model="qwen-max",
            messages=[
                {
                    'role':'system',
                    'content': content
                },
                {
                    'role': 'user',
                    'content': text
                }
            ],
        )
        output = completion.choices[0].message.content
        if isinstance(output, str):
            if '1' in output:
                return True
            if '0' in output:
                return False
        return None

    @classmethod
    def auto_fix_dialogue(cls, dialogue):
        content = f'你是一个对话纠错助手，你需要根据语音识别结果的输入，判断是否有错误的表达，然后进行修正。\n\
这是一个机场的自助值机系统，用户说话的内容大概率会和机场的服务相关，例如想要选择某一航班、或者选择某一座位等等，你进行纠错的范围要贴近机场的服务的主题。\n\
错误表达的例子有，用户想要去广州的航班，说的是”我想要去广州的飞机“，但是由于用户可能普通话不标准，导致识别的结果可能是“我想要去广州的北京”等错误表达。\n\
这里的错误基本上都是发音相似的字，例如上述错误就是“飞机”被错误识别成了“北京”，你要遵从用户的意思，得出对应发音最准确的文字，而不是用户想说“飞机”，你却改成了“航班”等意思近似但是发音差别很大的表达。\n\
用户表达正确的部分不需要进行修改，你只需要修正用户可能因为发音不标准而识别错误的部分。\n\
你要准确理解用户的意思，然后进行修正。给出正确且符合用户意思的表达。\n\
以下是需要修正的对话：\n{dialogue}\n请你输出修正后的对话。你只需要输出修正后的对话，不需要输出任何提示信息。'
        completion = cls.client.chat.completions.create(
            model="qwen-max",
            messages=[
                {
                    'role':'system',
                    'content': content
                },
                {
                    'role': 'user',
                    'content': dialogue
                }
            ],
        )
        output = completion.choices[0].message.content
        return output
