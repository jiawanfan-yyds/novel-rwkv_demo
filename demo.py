import gradio as gr
import os, sys, torch
import numpy as np
from gradio import components
np.set_printoptions(precision=4, suppress=True, linewidth=200)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
# Tune these below (test True/False for all of them) to find the fastest setting:
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
# torch._C._jit_override_can_fuse_on_cpu(True)
# torch._C._jit_override_can_fuse_on_gpu(True)
# torch._C._jit_set_texpr_fuser_enabled(False)
# torch._C._jit_set_nvfuser_enabled(False)

# set these before import RWKV
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '0' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

from rwkv.model import RWKV # pip install rwkv
model_default = './models/RWKV-4-Novel-7B-v1-Chn-20230426-ctx8192'
strategy_default = 'cuda fp16i8'
model = RWKV(model_default, strategy_default)

model_folder = "./models"  # 模型文件夹的路径
# 获取模型文件夹中的所有文件名
model_files = os.listdir(model_folder)
# 过滤出以 .pth 结尾的文件
model_files = [f for f in model_files if f.endswith(".pth")]
# 将文件名转换为选项格式
model_options = [os.path.splitext(os.path.basename(f))[0] for f in model_files]

out, state = model.forward([187, 510, 1563, 310, 247], None)
print(out.detach().cpu().numpy())                   # get logits
out, state = model.forward([187, 510], None)
out, state = model.forward([1563], state)           # RNN has state (use deepcopy to clone states)
out, state = model.forward([310, 247], state)
print(out.detach().cpu().numpy())                   # same result as above

from rwkv.utils import PIPELINE, PIPELINE_ARGS
pipeline = PIPELINE(model, "20B_tokenizer.json")

def generate_output(context, temperature, top_p, alpha_frequency, alpha_presence, token_count, model_name=os.path.basename(model_default), strategy=strategy_default):
    model_path = model_folder + "/" + os.path.splitext(model_name)[0]
    global model_default, strategy_default, model
    if model_path != model_default or strategy != strategy_default:
        model_default = model_path
        strategy_default = strategy
        model = RWKV(model=model_default, strategy=strategy_default)

    args = PIPELINE_ARGS(temperature=temperature,
                         top_p=top_p,
                         top_k=0, # top_k = 0 then ignore
                         alpha_frequency=alpha_frequency,
                         alpha_presence=alpha_presence,
                         token_ban=[0], # ban the generation of some tokens
                         token_stop=[], # stop generation whenever you see any token here
                         chunk_len=256) # split input into chunks to save VRAM (shorter -> slower)
    def my_print(s):
        print(s, end='', flush=True)

    pipeline_output = pipeline.generate(context, token_count, args=args, callback=my_print)
    return pipeline_output


novel = gr.Interface(fn=generate_output, 
                     inputs=[
                        gr.components.Textbox(lines=20,max_lines=20,label="输入",placeholder="输入部分只有最后约2000字生效"),                    
                        gr.components.Slider(label="Temperature", info="高则更发散更具创造性，低则更确切更严谨",minimum=0.1, maximum=2.5, step=0.1, default=1.0),
                        gr.components.Slider(label="Top_P", info="高则容易出现新颖的表达方式，低则倾向于已经出现的表达方式",minimum=0, maximum=1.0, step=0.05, default=0.7),
                        gr.components.Slider(label="countPenalty",info="越高则越少出现重复单词", minimum=0.1, maximum=1.0, step=0.05, default=0.5),
                        gr.components.Slider(label="presencePenalty",info="越高则越避免生成重复内容",minimum=0.1, maximum=1.0, step=0.05, default=0.5),
                        gr.components.Slider(label="Token Count",info="控制每次生成的文本长度",minimum=10, maximum=500, step=1, default=200),
                        gr.components.Dropdown(value=os.path.splitext(os.path.basename(model_default))[0], label="模型",choices=model_options),
                        gr.components.Dropdown(value=strategy_default,choices=[
                                                    "cuda fp16",
                                                    "cpu fp32",
                                                    "cuda fp16i8", 
                                                    "cuda fp16 *8 -> cpu fp32",
                                                    "cuda fp16 *6+",
                                                    "cpu fp32 *3 -> cuda fp16 *6+", 
                                                    "cuda fp16i8 *6 -> cuda fp16 *0+ -> cpu fp32 *1",                   
                                                    "cuda fp16 *0+ -> cpu fp32 *1",
                                                    "cuda:0 fp16 *20 -> cuda:1 fp16",
                                                    "cuda:0 fp16 -> cuda:1 fp16 -> cpu fp32 *1",],
                                           label="策略")
                     ],
                     outputs=[gr.components.Textbox(label="输出")],
                     examples=[
                            ["这赤色的雷霆落在孟浩身上，化作了无数的电光游走，可孟浩依旧站在半空，抬头间，气势不断地攀升。",1.3,0.65,0.5,0.5,200],
                            ["对于“创造一切的主”，他并不陌生，《风暴之书》《夜之启示录》和民俗传说里的那位造物主就有类似的称谓，极光会等隐秘组织信奉的“真实造物主”也被冠以相同的描述。但“全知全能的神”，克莱恩还是第一次在这个世界听说，不管黑夜女神，还是风暴之主，蒸汽与机械之神，都没有声称自己无所不知无所不能。",1.8,0.5,0.5,0.5,300]
                        ],
                     allow_flagging="never"   
                    )

strategy_cuda_cpu = ["cpu fp32","cup bf16","cpu fp32i8","cuda fp16","cuda fp16i8","cuda fp16i8 *20 -> cuda fp16","cuda fp16i8 *20+","cuda fp16i8 *20 -> cpu fp32 ","cuda:0 fp16 *20 -> cuda:1 fp16"]
config = ["内存：7B模型 需要32G","内存：7B模型 需要16G","内存：7B模型需要12G左右","显存：7B模型需要15G","显存：7B模型需要9G(编译CUDA可再省1-2G)","显存：需要9G-15G","显存：小于9G可用，但需要更多内存","显存：小于9G可用，但需要更多内存","将模型在两张显卡分配，可用两张卡的显存"]
speed = ["在intel速度较快，在amd非常慢","新intel（例如Xeon Platinum），支持bf16","省内存，速度比cpu fp32慢","速度快","省显存，速度较快","3B和7B模型有32+1层，14B模型有40+1层，前20层为fp16i8，后续层为cuda fp16","前20层为cuda fp16i8固定在GPU上，后续层在运算时再读入GPU","前20层为fp16i8固定在GPU上，后续层在CPU运算","前20层为fp16，在cuda:0，后续层在cuda:1"]

def strategyexample(strategy):
    index = strategy_cuda_cpu.index(strategy)
    return (config[index], speed[index])

strategy_about = gr.Interface(
    fn=strategyexample,
    inputs=gr.components.Radio(choices=strategy_cuda_cpu),
    outputs=[
        gr.components.Text(label = "内存显存占用情况"),
        gr.components.Text(label = "速度及详情")
                           ],
    interpretation="default",
    allow_flagging="never",
    live=True,
    description= "<b>[rmkv作者模型下载地址]<b>https://huggingface.co/BlinkDL \n<b>[rmkv作者关于strategy的详解]<b>https://zhuanlan.zhihu.com/p/609154637 \n<b>[本仓库地址]<b>"
)

demo = gr.TabbedInterface(interface_list=[novel,strategy_about],title = "基于rmkv模型的demo",tab_names=["续写小说","更多"])

if __name__ == "__main__":
    demo.launch(server_name="192.168.99.242",server_port=7777,share=True)
