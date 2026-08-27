Templates = {
    'base': "{task_template}",

    'llama2': "[INST] {task_template} [/INST]",

    'mistral_instruct': "{task_template}",

    'vicuna': "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {task_template} ASSISTANT:",

    'lwm-chat': "You are a helpful assistant. USER: {task_template} ASSISTANT: ",

    'command-r-chat': "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{task_template}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",

    'chatglm-chat': "[gMASK]sop<|user|> \n {task_template}<|assistant|> \n ",

    'RWKV': "User: hi\n\nAssistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it\n\nUser: {task_template}\n\nAssistant:",

    'Phi3': "<|user|>\n{task_template}<|end|>\n<|assistant|>\n",

    'llama3': "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{task_template}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",

    'qwen': "<|im_start|>user\n{task_template}<|im_end|>\n<|im_start|>assistant\n",

    'jamba': "<|startoftext|><|bom|><|system|> <|eom|><|bom|><|user|> {task_template}<|eom|><|bom|><|assistant|>",
}
