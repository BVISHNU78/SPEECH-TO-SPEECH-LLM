import os
import nltk
os.environ["CUDACXX"]="C:\\Program Files\\NVIDIA GPU Computing Toolkit\CUDA\\v11.0"
#os.system('python -m unidic download')
os.system('CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python==0.2.11 --verbose')
nltk.download('all')
from faster_whisper import WhisperModel
import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager
from utils import get_sentence,generate_speech_for_sentence, wave_header_chunk
import torch
print(torch.cuda.is_available()) 
print(torch.cuda.get_device_name(0))
whisper_model=WhisperModel("large-v3",device="cuda",compute_type="float32")
print("Loading whisper ASR")
local_dir = r"D:\coding\LLM\tts"
hf_hub_download(repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",local_dir=local_dir,filename="mistral-7b-instruct-v0.1.Q5_K_M.gguf")
mistral_model_path="D:/coding/LLM/mistral-7b-instruct-v0.1.Q5_K_M.gguf"
mistral_llm=Llama(model_path=mistral_model_path,n_gpu_layer=15,max_new_tokens=256,context_window=4096,n_ctx=4096,n_batch=128,verbose=False)
print("Loading XTTS model")
os.environ["COQUI_TOS_AGREED"]="1"
tts_model_name="tts_models/multilingual/multi-dataset/xtts_v2"
ModelManager().download_model(tts_model_name)
tts_model_path=os.path.join(get_user_data_dir(r"C:\Users\Dell\AppData\Local\tts"),tts_model_name.replace("/","--"))
config = XttsConfig()
config.load_json(os.path.join(tts_model_path,"config.json"))
xtts_model=Xtts.init_from_config(config)
xtts_model.load_checkpoint(
    config,
    checkpoint_path=os.path.join(tts_model_path,"model.pth"),
    vocab_path=os.path.join(tts_model_path,"vocab.json"),
    speaker_file_path=os.path.join(tts_model_path, "speakers_xtts.pth"),
    eval=True,
    use_deepspeed=False,
    )

xtts_model.cuda()

#grdio interface

with gr.Blocks(title="SPEECH TO LLM") as demo:
    DESCRIPTION="""# SPEECH TO LLM"""
    gr.Markdown(DESCRIPTION)

    chatbot=gr.Chatbot(
        value=[(None,"hello friend ,I am vishnu. How can I help you today?")],
        elem_id="Chatbot",
        avatar_images=("D:\\coding\\LLM\\speech to speech bot\\voice-bot-llm\\examples\\download.png","D:\\coding\LLM\\speech to speech bot\\voice-bot-llm\\examples\\download.png"),
        bubble_full_width=False,
  )

    VOICES = ["female", "male"]
    with gr.Row():
        chatbot_voice=gr.Dropdown(
            label="Voice of chatbot",
            info="How should chatbot talk like",
            choices=VOICES,
            max_choices=1,
            value=VOICES[1], 
    )

    with gr.Row():
        txt_box=gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter text and press enter or speak to your mic",
            container=False,
            interactive=True,
    )
    audio_record=gr.Audio(source="microphone",type="filepath",scale=4)

    with gr.Row():
        sentence=gr.Textbox(visible=False)
        audio_playback = gr.Audio(
            value=None,
            label=" Generated audio response",
            streaming=True,
            autoplay=True,
            interactive=False,
            show_label=True,
        )
        def add_text(chatbot_history,text):
            chatbot_history=[]if chatbot_history is None else chatbot_history
            chatbot_history=chatbot_history + [(text,None)]
            return chatbot_history,gr.update(value="",interactive=False)
        def add_audio(chatbot_history,audio):
            chatbot_history=[] if chatbot_history is None else chatbot_history
            response,_=whisper_model.transcribe(audio)
            text = list(response)[0].text.strip()
            print("Transcribed text:",text)
            chatbot_history=chatbot_history +[(text,None)]
            return chatbot_history,gr.update(value="",interactive=False)
        def generate_speech(chatbot_history, chatbot_voice, initial_greeting=False):
            yield("",chatbot_history,wave_header_chunk())
            def handle_speech_generation(sentence,chatbot_history,chatbot_voice):
                if sentence !="":
                    print("sentence processing")
                    generated_speech=generate_speech_for_sentence(chatbot_history,chatbot_voice,sentence,xtts_model,xtts_supported_languages=config.languages,return_as_byte=True)
                    if generated_speech is not None:
                        _,audio_dict =generated_speech
                        yield(sentence,chatbot_history,audio_dict['value'])
            if initial_greeting:
                for _,sentence in chatbot_history:
                    yield from handle_speech_generation(sentence,chatbot_history,chatbot_voice)
            else:
                for sentence,chatbot_history in get_sentence(chatbot_history,mistral_llm):
                    print("Inserting sentence to queue")
                    yield from handle_speech_generation(sentence,chatbot_history,chatbot_voice)                                    
    txt_msg = txt_box.submit(fn=add_text, inputs=[chatbot, txt_box], outputs=[chatbot, txt_box], queue=False
                             ).then(fn=generate_speech,  inputs=[chatbot,chatbot_voice], outputs=[sentence, chatbot, audio_playback])
    txt_msg.then(fn=lambda: gr.update(interactive=True), inputs=None, outputs=[txt_box], queue=False)
    audio_msg = audio_record.stop_recording(fn=add_audio, inputs=[chatbot, audio_record], outputs=[chatbot, txt_box], queue=False
                                            ).then(fn=generate_speech,  inputs=[chatbot,chatbot_voice], outputs=[sentence, chatbot, audio_playback])
    audio_msg.then(fn=lambda: (gr.update(interactive=True),gr.update(interactive=True,value=None)), inputs=None, outputs=[txt_box, audio_record], queue=False)
   
    demo.load(fn=generate_speech,inputs=[chatbot,chatbot_voice,gr.State(value=True)],outputs=[sentence,chatbot,audio_playback])
demo.queue().launch(debug=True,share=True)