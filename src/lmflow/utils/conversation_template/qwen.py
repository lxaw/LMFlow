#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
from typing import Dict, Set, Sequence, Literal, Union, List, Optional, Tuple

from transformers import PreTrainedTokenizer

from .base import StringFormatter, TemplateComponent, ConversationTemplate, ConversationTemplateForTool


QWEN2_TEMPLATE = ConversationTemplate(
    template_name='qwen2',
    user_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|im_start|>user\n{{content}}<|im_end|>\n')
        ]
    ),
    assistant_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|im_start|>assistant\n{{content}}<|im_end|>\n')
        ]
    ),
    system_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|im_start|>system\n{{content}}<|im_end|>\n')
        ]
    ),
    separator=TemplateComponent(type='string', content='\n')
)


QWEN2_TEMPLATE_FOR_TOOL = ConversationTemplateForTool(
    template_name='qwen2_for_tool',
    user_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|im_start|>user\n{{content}}<|im_end|>\n')
        ]
    ),
    function_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|im_start|>assistant\n{{content}}<|im_end|>\n')
        ]
    ),
    observation_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|im_start|>tool\n{{content}}<|im_end|>\n')
        ]
    ),
    assistant_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|im_start|>assistant\n{{content}}<|im_end|>\n')
        ]
    ),
    system_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|im_start|>system\n{{content}}<|im_end|>\n')
        ]
    ),
    separator=TemplateComponent(type='string', content='\n')
)


class Qwen2_5ConversationTemplate(ConversationTemplateForTool):
    def _handle_tools(self, tools: Optional[List[str]]) -> str:
        prompt1 = "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        prompt2 = '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
        tools_out = ''
        
        if tools is not None:
            tools_out = prompt1
            for tool in tools:
                tools_out += "\n" + tool
            tools_out += prompt2
        
        return tools_out
    

QWEN2_5_TEMPLATE = Qwen2_5ConversationTemplate(
    template_name='qwen2_5',
    user_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|im_start|>user\n{{content}}<|im_end|>\n')
        ]
    ),
    function_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|im_start|>assistant\n{{content}}<|im_end|>\n')
        ]
    ),
    observation_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|im_start|>tool\n{{content}}<|im_end|>\n')
        ]
    ),
    assistant_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|im_start|>assistant\n{{content}}<|im_end|>\n')
        ]
    ),
    system_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|im_start|>system\n{{content}}<|im_end|>\n')
        ]
    ),
    separator=TemplateComponent(type='string', content='\n'),
    system_default='You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.'
)