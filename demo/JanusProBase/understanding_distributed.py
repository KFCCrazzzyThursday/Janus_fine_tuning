import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import PIL.Image
import numpy as np

from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
def ddp_setup():
    """
    初始化分布式进程组，并设置当前进程使用哪张GPU。
    LOCAL_RANK/ RANK / WORLD_SIZE 都是 torchrun 自动传入的环境变量
    """
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])  # 当前进程在本机的 GPU 编号
    torch.cuda.set_device(local_rank)           # 让本进程只用 local_rank 对应的 GPU
    return local_rank
@torch.inference_mode()
def generate_images(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1.0,
    parallel_size: int = 2,
    cfg_weight: float = 5.0,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    """
    与之前的单卡逻辑类似，但要注意我们在外部已经把模型包装成 DDP，
    因此如果 mmgpt 是 DDP 对象，需要访问底层模型 (mmgpt.module)。
    另外我们可以在 batch 维度并行，从而让多卡分摊计算。
    """
    is_ddp = isinstance(mmgpt, DDP)
    if is_ddp:
        real_model = mmgpt.module
    else:
        real_model = mmgpt

    # 把 prompt 编码成 tokens
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids).cuda()

    # 构建一次性 batch = parallel_size*2，用于 CFG（cond/uncond）
    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    # 得到输入的 embeddings
    inputs_embeds = real_model.language_model.get_input_embeddings()(tokens)
    # 准备存放生成的图像tokens
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

    past_key_values = None
    for step_i in range(image_token_num_per_image):
        # 喂给底层语言模型
        outputs = real_model.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=past_key_values
        )
        hidden_states = outputs.last_hidden_state
        past_key_values = outputs.past_key_values

        # 执行 CFG
        logits = real_model.gen_head(hidden_states[:, -1, :])
        logits_cond = logits[0::2, :]
        logits_uncond = logits[1::2, :]
        logits = logits_uncond + cfg_weight * (logits_cond - logits_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)  # [parallel_size, 1]
        generated_tokens[:, step_i] = next_token.squeeze(dim=-1)

        # 把 cond / uncond 拼回
        next_token = torch.cat([next_token, next_token], dim=1).view(-1)

        # 准备下一步图像embedding
        img_embeds = real_model.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    # 解码为图像
    dec = real_model.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size]
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    # 转成可保存格式
    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    os.makedirs('generated_samples_2', exist_ok=True)
    for i in range(parallel_size):
        save_path = os.path.join('generated_samples_2', f"rank{dist.get_rank()}_img_{i}.jpg")
        PIL.Image.fromarray(visual_img[i]).save(save_path)

    print(f"Rank{dist.get_rank()} is done. Images are saved to generated_samples/")

def main():
    local_rank = ddp_setup()

    # ----------------------------------------------------------
    # （1）加载模型和处理器
    # ----------------------------------------------------------
    model_path = "deepseek-ai/Janus-Pro-7B"
    cache_dir  = "/workspace/liyj/Janus_fine_tuning/ckpt/JanusPro7B"

    print(f"[Rank {dist.get_rank()}] is loading Base model...")

    vl_chat_processor = VLChatProcessor.from_pretrained(model_path, cache_dir=cache_dir)
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, cache_dir=cache_dir
    )

    # 转到本进程使用的那张 GPU
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda(local_rank)
    vl_gpt.eval()

    # ----------------------------------------------------------
    # （2）构建 DDP 包装
    # ----------------------------------------------------------
    vl_gpt = DDP(vl_gpt, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # ----------------------------------------------------------
    # （3）准备对话 / prompt
    # ----------------------------------------------------------
    question = '''
    "question":"Which of the following could Gordon's test show?",
        "choices":[
          "if the spacecraft was damaged when using a parachute with a 1 m vent going 200 km per hour",
          "how steady a parachute with a 1 m vent was at 200 km per hour",
          "whether a parachute with a 1 m vent would swing too much at 400 km per hour"
        ],
        "answer":1,
        "hint":"People can use the engineering-design process...",
        "skill":"Evaluate tests of engineering-design solutions",
        "lecture":"People can use the engineering-design process..."
    '''
    conversation = [
        {
            "role": "<|User|>",
            "content": "Generate an image compatible for this QA pair:",
        },
        {"role": "<|Assistant|>", "content": f"{question}"},
    ]

    # 应用 SFT 模板
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt=""
    )
    prompt = sft_format + vl_chat_processor.image_start_tag

    # ----------------------------------------------------------
    # （4）调用生成函数
    #     这里设置 parallel_size=2 以便有batch维度让两张卡都能并行
    # ----------------------------------------------------------
    print(f"[Rank {dist.get_rank()}] generation starts...")
    generate_images(
        mmgpt=vl_gpt,
        vl_chat_processor=vl_chat_processor,
        prompt=prompt,
        temperature=1.25,
        parallel_size=1,
        cfg_weight=2.5,
        image_token_num_per_image=576,
        img_size=384,
        patch_size=16,
    )

    # ----------------------------------------------------------
    # （5）销毁进程组，结束
    # ----------------------------------------------------------
    torch.cuda.synchronize()
    dist.destroy_process_group()

if __name__ == "__main__":
    # 使用 torchrun --nproc_per_node=2 understanding_distributed.py 启动，此处会并行启动2个进程
    main()
