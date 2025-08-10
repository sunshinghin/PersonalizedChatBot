

#################################################################################################
#   For people who is willing to read the codes.                                              #
#   I code this thing mostly at around 12 - 3am.                                                #
#   If you find out my code is bad or the organisation of the code the bad,I am sorry for that. #
#   Don't judge me too hard please!!!                                                           #
#################################################################################################

import os

#checking if the required libraries are installed

try:
    import ollama
    import flask
    import wtforms
    import pydantic
    import json
    import subprocess
    import piper
    import wave
    import datetime
    import playsound
    import webbrowser
    import RealtimeSTT
    import pdfplumber
    import mss
except ImportError:
    print("Required libraries are not installed. Installing now...")
    os.system("pip install -r requirements.txt")
    
#end

import flask
from flask import render_template, request, redirect,Flask,jsonify,url_for
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, IntegerField, SelectField, MultipleFileField, StringField, TextAreaField,MultipleFileField,BooleanField
from wtforms import validators
from wtforms.validators import InputRequired
from ollama import chat
from pydantic import BaseModel
import json
import subprocess
import wave
from piper import PiperVoice
import datetime
from playsound import playsound
import torch
import pyaudio
from faster_whisper import WhisperModel
import time
import gc
import threading
from threading import Thread
import webbrowser
from RealtimeSTT import AudioToTextRecorder
import pdfplumber
import mss
import win32api
import win32con

##GLOBAL VAR AND FUNCTION

SUPPORTED_IMG_EXTENSIONS = [".jpg",".jpeg",".png"]
SUPPORTED_DOCUMENT_EXTENSIONS = [".pdf"]

RUN_HOST = "127.0.0.1"
RUN_PORT = 5000

locks_for_prev_request = threading.Lock()

device = "cuda" if torch.cuda.is_available() else "cpu"

vision_model = "qwen2.5vl:3b"
thinking_model = "gemma3:4b"

recording = False
chunk = 1024
format = pyaudio.paInt16
channels = 2
rate = 44100
STT_write_wav_filepath = ""
transcripted = ""
recording_saved = False
transcripted_done = True

realtime_stt = False
realtime_STT_recorder = ""

chatnumber = 0

def record(FILE_NAME):

    global recording_saved
    global STT_write_wav_filepath
    global recording

    STT_write_wav_filepath = app.config["VOICEINPUT_FOLDER"] + "/" + FILE_NAME

    p = pyaudio.PyAudio()

    stream = p.open(format=format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    frames = []
    
    print("Recording...")

    while recording == True:
        
        data = stream.read(chunk)  
        frames.append(data)
        

    print("Finished recording... saving")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(STT_write_wav_filepath, 'wb')  
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    recording_saved = True

#ends

#setup and check for json file

saved_data = {}
valid_data = False

with open("./log.json", "r", encoding="utf-8") as f:
    try:
        
        saved_data = json.load(f)

        valid_data = True

    except json.JSONDecodeError:

        saved_data = {}
        valid_data = False
        

with open("./log.json", "w", encoding="utf-8") as f:
    if valid_data == False:
        #if the file is empty or invalid, we will write an empty dict to the file
        json.dump({}, f)
    else:
        #if the file is valid, we will write the saved data back to the file
        json.dump(saved_data, f)


with open("./profile.json", "r", encoding="utf-8") as f:
    try:
        
        saved_data = json.load(f)

        check_data = saved_data["character_name"]
        check_data = saved_data["mood"]
        check_data = saved_data["chatnumber"]
        chatnumber = check_data
        check_data = saved_data["thinking_model_name"]
        check_data = saved_data["vision_model_name"]
        check_data = saved_data["character_desc"]
        check_data = saved_data["thought_about_user"]
        check_data = saved_data["TTS"]
        check_data = saved_data["TTS_model_name"]
        check_data = saved_data["Long-term_Memory"]
        check_data = saved_data["User_name"]
        check_data = saved_data["STT_model_name"]
        check_data = saved_data["Realtime_STT_model_name"]

        if __name__ == "__main__":

            realtime_STT_recorder = AudioToTextRecorder(model=check_data,device=device,compute_type="float16" if torch.cuda.is_available() else "float32")

        valid_data = True



    except json.JSONDecodeError or KeyError:
        
        #if the file is empty or invalid, we will write an empty dict to the file
        saved_data = {}
        valid_data = False


with open("./profile.json", "w", encoding="utf-8") as f:
    if valid_data == False:
        json.dump({"character_name": thinking_model, "mood": "Neutral", "chatnumber": 0, "thinking_model_name": thinking_model, "character_desc": "", "thought_about_user": "","TTS": False,"TTS_model_name": "en_US-sevenseven-medium.onnx","User_name": "User","STT_model_name": "medium.en","vision_model_name": vision_model,"Realtime_STT_model_name": "tiny.en","Long-term_Memory": ""}, f)
    
        chatnumber = 0

    else:
        #if the file is valid, we will write the saved data back to the file
        json.dump(saved_data, f)

#check if required model is installed

def check_if_model_downloaded():

    cmd_result = subprocess.run(["ollama","list"], stdout=subprocess.PIPE).stdout.decode("utf-8")

    with open("./profile.json", "r", encoding="utf-8") as f:
        try:

            profile = json.load(f)

            saved_thinking_model_name = profile["thinking_model_name"]
            saved_vision_model_name = profile["vision_model_name"]

            if saved_thinking_model_name not in cmd_result:

                os.system("ollama pull " + saved_thinking_model_name)

                print("Downloading thinking model: " + saved_thinking_model_name)

            if saved_vision_model_name not in cmd_result:

                os.system("ollama pull " + saved_vision_model_name)

                print("Downloading vision model: " + saved_vision_model_name)

        except json.JSONDecodeError:

            raise ValueError("Cannot find the model name of the chatbot saved.")

check_if_model_downloaded()

# check end

#app config list

app = Flask(__name__)
app.config['SECRET_KEY'] = "mysecretkeylol"
app.config['UPLOAD_FOLDER'] = './UploadFiles'
app.config['CUSTOMVOICEMODELS_FOLDER'] = "./CustomVoiceModels"
app.config["VOICEMODELSOUTPUT_FOLDER"] = "./output"
app.config["VOICEINPUT_FOLDER"] = "./input"
app.config["SOUNDEFFECTS_FOLDER"] = "./SoundEffects"

#end

class SettingsForm(FlaskForm):

    character_name_input = StringField(label="CharacterName",default="",description="Input your character name here",render_kw={"placeholder": "Enter your character name","class": "form-control","size": "60"})
    user_name_input = StringField(label="UserName",default="",description="Input your name here",render_kw={"placeholder": "Enter your name","class": "form-control","size": "60"})
    thinking_model_name_input = SelectField(label="ThinkingModelName",choices=[("gemma3:4b","gemma3:4b"),("deepseek-r1:7b","deepseek-r1:7b"),("llama3:8b","llama3:8b"),("llama3:70b","llama3:70b")],default="gemma3:4b",description="Select your thinking model name",render_kw={"class": "form-select"})
    vision_model_name_input = SelectField(label="VisionModelName",choices=[("qwen2.5vl:3b","qwen2.5vl:3b"),("gemma3:4b","gemma3:4b"),("llama3.2-vision:11b","llama3.2-vision:11b")],default="qwen2.5vl:3b",description="Select your vision model name",render_kw={"class": "form-select"})
    character_desc_input = TextAreaField(label="CharacterDescription",default="",description="Input your character description here",render_kw={"placeholder": "Enter your character description","class": "form-control"})

    Submit = SubmitField(label="Submit",render_kw={"class": "btn btn-primary"})
    ClearBtn = SubmitField(label="Clear Chat Log",render_kw={"class": "btn btn-danger"})
    ClearFilesBtn = SubmitField(label="Clear Uploaded Files",render_kw={"class": "btn btn-danger"})

    TTS_onoff = BooleanField(label="TTS On/Off",default=False,render_kw={"class": "form-check-input","role": "switch","id": "switchCheckDefault"})
    TTS_model_name_input = StringField(label="TTSModelName",default="",description="Input your TTS model name here",render_kw={"placeholder": "Enter your TTS model name","class": "form-control","size": "60"})

    Clear_TTS_wavs_Btn = SubmitField(label="Clear TTS sound files",render_kw={"class": "btn btn-danger"})

    STT_model_name_input = SelectField(label="STTModelName",choices=[("small.en","small.en"),("medium.en","medium.en"),("large-v3","large-v3")],default="medium.en",description="Select your STT model name",render_kw={"class": "form-select"})
    Clear_STT_wavs_Btn = SubmitField(label="Clear STT sound files",render_kw={"class": "btn btn-danger"})

    realtime_STT_model_name_input = SelectField(label="STTModelName",choices=[("tiny.en","tiny.en"),("small.en","small.en"),("medium.en","medium.en"),("large.en","large.en")],default="tiny.en",description="Select your Realtime STT model name",render_kw={"class": "form-select"})

class Chatform(FlaskForm):

    InputChat = StringField(label="InputChat",description="Input your chat here",render_kw={"placeholder": "Chat here...","class": "form-control", "size": "102"})
    Submit = SubmitField(label="Submit",render_kw={"class": "btn btn-primary"})
    FileUploadBox = MultipleFileField(label="FileUpload",render_kw={"class": "form-control", "id": "formFile"},default=None)
    STTbtn = SubmitField(label="Record",render_kw={"class": "btn btn-success","onclick": "recordbtn()","id": "STTbtn"})

    Realtime_STT_Btn = SubmitField(label="Enable Realtime STT",render_kw={"class": "btn btn-success","onclick": "realtimeSTTBtnOnclick()","id": "Realtime_STT_btn"})

class StructuredResponse(BaseModel):
    mood: str
    response_to_user: str
    thought_about_user: str
    website_to_open: str

@app.route('/', methods=['GET', 'POST'])

def index_page():

    global chatnumber
    global recording
    global STT_write_wav_filepath
    global transcripted
    global recording_saved
    global transcripted_done
    global realtime_stt
    global realtime_STT_recorder

    chatform = Chatform()

    #declare the variables for global use in this function

    write_dict = {}
    character_name = ""
    mood = ""
    character_desc = ""
    thought_about_user = ""
    TTS_option = False
    TTS_model_name = ""
    user_name = ""
    STT_model = ""
    thinking_model = ""
    vision_model = ""
    Memory = ""

    output = ""
    #used for when user load in the website, load the chat log

    with open("./log.json", "r", encoding="utf-8") as f:

        json_saved = f.read()

        try:
            #load the json file back to a dict
            write_dict = json.loads(json_saved)
        except json.JSONDecodeError:
            write_dict = {}

    with open("./profile.json", "r", encoding="utf-8") as f:
        try:
            #load the profile file back to a dict
            profile = json.load(f)
            character_name = profile["character_name"]
            mood = profile["mood"]
            character_desc = profile["character_desc"]
            thinking_model = profile["thinking_model_name"]
            vision_model = profile["vision_model_name"]
            thought_about_user = profile["thought_about_user"]
            TTS_option = profile["TTS"]    
            TTS_model_name = profile["TTS_model_name"]
            user_name = profile["User_name"]
            STT_model = profile["STT_model_name"]
            Memory = profile["Long-term_Memory"]

        except json.JSONDecodeError:
            profile = {}
            character_name = thinking_model
            mood = "Neutral"
            character_desc = ""
            thinking_model = "gemma3:4b"
            vision_model = "qwen2.5vl:3b"
            thought_about_user = ""
            TTS_option = False
            TTS_model_name = "en_US-sevenseven-medium.onnx"
            user_name = "User"
            STT_model = "medium.en"
            Memory = ""

    #when user click the submit button

    waking = False

    can_request = False

    if chatform.validate_on_submit():

        can_request = True
    
        if chatform.Realtime_STT_Btn.data and locks_for_prev_request.locked() == False:

            if realtime_stt == True:

                realtime_stt = False

            else:

                realtime_stt = True


    else:

        can_request = False

    if realtime_stt == True:

        can_request = True


    if can_request and locks_for_prev_request.locked() == False:
        #STT
        locks_for_prev_request.acquire()
        raw_input = chatform.InputChat.data

        #REALTIME STT TEST

        if realtime_stt == True:

            

            start_notification_soundpath = os.path.join(app.config["SOUNDEFFECTS_FOLDER"],"start_notification.mp3")
            end_notification_soundpath = os.path.join(app.config["SOUNDEFFECTS_FOLDER"],"end_notification.mp3")
            ready_notification_soundpath = os.path.join(app.config["SOUNDEFFECTS_FOLDER"],"ready_notification.mp3")

            realtime_transcript = ""
            waking = False

            #recorder = AudioToTextRecorder(model=realtime_STT_model_name,device=device,compute_type="float16" if torch.cuda.is_available() else "float32")

            recorder = realtime_STT_recorder

            playsound(ready_notification_soundpath,block=False)

            while waking == False and realtime_stt == True:

                realtime_transcript = recorder.text().lower()

                print(realtime_transcript)

                if character_name.lower() in realtime_transcript:
                                    
                    waking = True

            playsound(start_notification_soundpath,block=False)

            realtime_transcript = recorder.text()

            playsound(end_notification_soundpath,block=False)

            if realtime_stt == False:

                return render_template('index.html',chat_log=write_dict,chatform=chatform,character_name=character_name,mood=mood,user_name=user_name,output=output,realtime_stt=realtime_stt,request_lock=locks_for_prev_request.locked())

            print("STT REAL TRANSCRIPT: " + realtime_transcript)

            raw_input = realtime_transcript

            print("Raw input: " + raw_input)

        #end

        if chatform.STTbtn.data:

            if recording == False:

                print("starting recording session")

                transcripted_done = False

                recording = True
                recording_saved = False
                
                record(str(chatnumber) + "_in.wav")

            else:

                print("stoppping recording session")

                recording = False

                #wait until the recording is saved

                while recording_saved == False:

                    time.sleep(0.5)

                    print("waiting for save")

                print("file path:" + STT_write_wav_filepath)

                STT_model = WhisperModel(STT_model,device=device,compute_type="float16" if torch.cuda.is_available() else "float32")

                segments, info = STT_model.transcribe(STT_write_wav_filepath)
                for segment in segments:

                    transcripted = segment.text

                    #print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

                print("Transcript: " + transcripted)

                raw_input = transcripted

                del STT_model

                gc.collect()

                transcripted_done = True

        #end of STT

        #LLM PART

        #MEMORY AND EMOTION PART

        #transcript done default is set to True when user only use text input but set the false when the transcript for STT is not done

        if transcripted_done == True:

            img_filepaths = []
            document_filepaths = []

            if chatform.FileUploadBox.data[0].filename != "":
                
                print("there is file and renaming the file")

                for file in chatform.FileUploadBox.data:
                    
                    file_fullname = file.filename
                    
                    _,file_extension = os.path.splitext(file_fullname)
                    #filename = os.path.split(file_fullname)[0]

                    print("Detcted file ext: " + file_extension)

                    filepath = os.path.join(app.config['UPLOAD_FOLDER'],str(chatnumber) + "_file" + str(file_extension))
                    file.save(filepath)

                    if file_extension in SUPPORTED_IMG_EXTENSIONS:

                        img_filepaths.append(filepath)

                    elif file_extension in SUPPORTED_DOCUMENT_EXTENSIONS:

                        document_filepaths.append(filepath)

                    else:

                        print("File is not supported! Filename: " + str(file_fullname))

            

            chat_log = {}

            with open("./log.json", "r", encoding="utf-8") as f:

                json_saved = f.read()

                try:
                    #load the json file back to a dict
                    chat_log = json.loads(json_saved)
                except json.JSONDecodeError:
                    chat_log = {}

            reverse_chat_log = dict(reversed(chat_log.items()))

            asked_questions_memoery = ""

            questions_idx = 0

            #read asked question from chat log

            for chatnum,chatdata in reverse_chat_log.items():

                if questions_idx > 3:

                    break

                saved_chatdata_answer = ""
                saved_chatdata_question = ""
                saved_chatdata_time = ""

                try:

                    saved_chatdata_question = chatdata["Question"]
                    saved_chatdata_answer = chatdata["Answer"]

                except KeyError:

                    saved_chatdata_question = ""
                    saved_chatdata_answer = ""

                try:

                    saved_chatdata_time = chatdata["Time"]

                except KeyError:

                    saved_chatdata_time = ""

                asked_questions_memoery = asked_questions_memoery + "Question" + str(chatnum) + ": " + saved_chatdata_question + " " + "Answer" + str(chatnum) + ": " + saved_chatdata_answer + " " + "Date: " + str(saved_chatdata_time) + ","

                questions_idx += 1
            

            with open("./profile.json", "r", encoding="utf-8") as f:

                try:
                    #load the profile file back to a dict
                    profile = json.load(f)
                    character_name = profile["character_name"]
                    mood = profile["mood"]
                    character_desc = profile["character_desc"]
                    thinking_model = profile["thinking_model_name"]
                    vision_model = profile["vision_model_name"]
                    thought_about_user = profile["thought_about_user"]
                    TTS_option = profile["TTS"]
                    TTS_model_name = profile["TTS_model_name"]
                    user_name = profile["User_name"]
                    STT_model = profile["STT_model_name"]
                    Memory = profile["Long-term_Memory"]

                except json.JSONDecodeError:
                    profile = {}
                    character_name = thinking_model
                    mood = "Neutral"
                    character_desc = ""
                    thinking_model = "gemma3:4b"
                    vision_model = "qwen2.5vl:3b"
                    thought_about_user = ""
                    TTS_option = False
                    TTS_model_name = "en_US-sevenseven-medium.onnx"
                    user_name = "User"
                    STT_model = "medium.en"
                    Memory = ""

            #END FOR MEMORY AND EMOTION PART

            #action test

            action_reponse = chat(messages=[{

                "role": "user",
                "content": f"This is a coversation between two people.What actions you think it needs to take to answer this question: {raw_input}? Only answer in these options: take computer screenshot,just chatting,need to use realtime newspaper.",

            }],
            
                model=thinking_model,

            )

            print("Action should take: " + action_reponse["message"]["content"])

            action = action_reponse["message"]["content"]
            ss_filepath = []
            ss_append_path = ""


            if "newspaper" in action.lower():

                webbrowser.open("https://edition.cnn.com/?hpt=header_edition-picker",new=2)

                win32api.SetCursorPos((100,500))

                #wait for the website to load
                time.sleep(10)

                win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, -1 * 340, 0)

                time.sleep(3)

                ss_append_path = os.path.join(app.config["UPLOAD_FOLDER"],str(chatnumber) + "_ss_1.png")

                ss_filepath.append(ss_append_path)

                with mss.mss() as sct:

                    monitor = {'left': 0, 'top': 60, 'width': 1920, 'height': 900}

                    sct_image = sct.grab(monitor)

                    mss.tools.to_png(sct_image.rgb,sct_image.size,output=ss_append_path)

                time.sleep(0.5)

                win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, -1 * 1040, 0)

                time.sleep(3)

                ss_append_path = os.path.join(app.config["UPLOAD_FOLDER"],str(chatnumber) + "_ss_2.png")

                ss_filepath.append(ss_append_path)

                with mss.mss() as sct:

                    monitor = {'left': 0, 'top': 60, 'width': 1920, 'height': 900}

                    sct_image = sct.grab(monitor)

                    mss.tools.to_png(sct_image.rgb,sct_image.size,output=ss_append_path)

                time.sleep(0.5)

            if "screenshot" in action.lower():

                ss_append_path = os.path.join(app.config["UPLOAD_FOLDER"],str(chatnumber) + "_ss.png")

                ss_filepath.append(ss_append_path)

                with mss.mss() as sct:

                    monitor = {'left': 0, 'top': 60, 'width': 1920, 'height': 900}

                    sct_image = sct.grab(monitor)

                    mss.tools.to_png(sct_image.rgb,sct_image.size,output=ss_append_path)


            #action test end

            time_now = str(datetime.datetime.now())

            images_desc = ""
            document_number = 1
            document_desc = ""

            #due with document part

            if document_filepaths != []:

                for filepath in document_filepaths:

                    _,file_extension = os.path.splitext(filepath)

                    if file_extension == ".pdf":

                        text = ""

                        with pdfplumber.open(filepath) as pdf_file:

                            for page in pdf_file.pages:

                                text = text + page.extract_text() + "\n"

                        document_desc = document_desc + "Document" + str(document_number) + ": " + text + ","

            print("Processed document desc: " + document_desc)

            #due with images part

            if img_filepaths != [] or ss_filepath != []:

                print("Generating Image desc...")

                #If user uploaded files,use the files and ignore screenshot

                gen_desc_objs = img_filepaths if img_filepaths != [] else ss_filepath

                images_desc = chat(messages=[{

                    "role": "user",
                    "content": "Describe the images under 50 words and read the text in the images",
                    "images": gen_desc_objs, 

                }],

                    model=vision_model

                )

                images_desc = images_desc["message"]["content"]

            print("Image desc: " + images_desc)

            input_chat = f"""

                You are going to role play as a person named {character_name} with the following description: {character_desc}.And you are in the mood of {mood} and the name of the user is {user_name}.You are going to have a casual conversation between you and the user.

                The time now is: {time_now}.

                Here is the long-term memory: {Memory}.

                Here are the questions and answers in the chat history: {asked_questions_memoery} 

                Here is the your thought about the user: {thought_about_user}.

                Here are the description of the images: {images_desc}.

                Here are the text of all the documents: {document_desc}.

                Now, answer this question only in text and under 25 words: {chatform.InputChat.data}.

                You can change your mood and thought based on the question and answer.

                Return the a website link in the website_to_open variable when the user's action is needed to open a website.Otherwise,just return empty string.

                Only return your mood,your response and your thought about the user and website to open.

            """

            print("Generating response...")

            response = chat(messages=[{

                "role": "user",
                "content": input_chat,
                
            }],
            

                model=thinking_model,
                format=StructuredResponse.model_json_schema(),

            )

            print("raw output: " + response["message"]["content"])

            Structuredoutput = StructuredResponse.model_validate_json(response["message"]["content"])

            #update the mood and thought about user

            mood = Structuredoutput.mood

            thought_about_user = Structuredoutput.thought_about_user

            website_to_open = Structuredoutput.website_to_open

            print("Generate successfully!")

            print("Generating Long-term memory...")

            #for long-term memory

            response = chat(messages=[{

                "role": "user",
                "content": f"Summarize the following conversation, highlighting key points and conclusions and include the time: {time_now} in under 40 words.Previous Summary: {Memory}.Conversation: User: {raw_input},AI: {Structuredoutput.response_to_user}",
                
            }],
            

                model=thinking_model,

            )

            long_term_memory = response["message"]["content"]

            #end

            print("Summary memory: " + long_term_memory)

            prev_chat_log = {}

            with open("./log.json", "r", encoding="utf-8") as f:

                try:
                    #load the prev log file back to a dict
                    prev_chat_log = json.load(f)
                except json.JSONDecodeError:
                    prev_chat_log = {}

            #update the chat log with the new question and answer

            date_now = str(datetime.datetime.now())

            prev_chat_log[chatnumber] = {"Question": raw_input, "Answer": Structuredoutput.response_to_user, "Time": date_now}

            write_dict = prev_chat_log

            with open("./log.json", "w",encoding="utf-8") as f:

                json.dump(write_dict,f)

            chatnumber += 1

            #we dont want to change the char name and char desc and model name in the profile.json file, so we will read the profile.json file first and then write it back with the new values

            prev_profile = {}

            with open("./profile.json", "r", encoding="utf-8") as f:
                try:
                    #load the profile file back to a dict
                    prev_profile = json.load(f)
                except json.JSONDecodeError:
                    prev_profile = {}
                    raise ValueError("Profile file is empty or corrupted. Please check the profile.json file.")
            

            prev_profile["mood"] = mood
            prev_profile["chatnumber"] = chatnumber
            prev_profile["thought_about_user"] = thought_about_user
            prev_profile["Long-term_Memory"] = long_term_memory

            write_profile_dict = prev_profile

            with open("./profile.json", "w", encoding="utf-8") as f:
                json.dump(write_profile_dict, f)

            if TTS_option == True:

                #TTS FEATURE

                print("Generating TTS...")

                TTS_model_name_filepath = app.config["CUSTOMVOICEMODELS_FOLDER"] + "/" + TTS_model_name

                voice = PiperVoice.load(TTS_model_name_filepath)

                wav_output_filepath = app.config["VOICEMODELSOUTPUT_FOLDER"] + "/" + str(chatnumber) + "_output.wav"

                with wave.open(wav_output_filepath, "wb") as wav_file:

                    voice.synthesize_wav(text=Structuredoutput.response_to_user,wav_file=wav_file)

                playsound(wav_output_filepath, block=False)

                print("Generated TTS")

                #end tts feature

            print("website to open: " + website_to_open)

            if website_to_open != "" and website_to_open != "nothing":

                print("opening website")

                webbrowser.open(website_to_open,new=2)

            locks_for_prev_request.release()

            if realtime_stt == True:

                return render_template('index.html',chat_log=write_dict,chatform=chatform,character_name=character_name,mood=mood,user_name=user_name,output=output,realtime_stt=realtime_stt,request_lock=locks_for_prev_request.locked())
            

    elif locks_for_prev_request.locked():

        output = "Warning: Please wait for the previous generation to be done."

    else:
        output = ""

    chatform.InputChat.data = ""

    return render_template('index.html',chat_log=write_dict,chatform=chatform,character_name=character_name,mood=mood,user_name=user_name,output=output,realtime_stt=realtime_stt,request_lock=locks_for_prev_request.locked())

@app.route('/settings.html', methods=['GET', 'POST'])
def settings_page():

    global realtime_STT_recorder

    #load profile.json for saved char desc,char name and etc.
    settingform = SettingsForm()
    output_message = ""

    profile_dict = {}

    with open("./profile.json", "r", encoding="utf-8") as f:

        try:

            profile_dict = json.load(f)

        except json.JSONDecodeError:
            output_message = "ErrorDECODE: profile.json is empty or corrupted"
            return render_template('settings.html', settingform=settingform,output=output_message)

    saved_character_name = profile_dict["character_name"]
    saved_character_desc = profile_dict["character_desc"]
    saved_thinking_model_name = profile_dict["thinking_model_name"]
    saved_vision_model_name = profile_dict["vision_model_name"]
    saved_tts_option = profile_dict["TTS"]
    saved_tts_model_name = profile_dict["TTS_model_name"]
    saved_user_name = profile_dict["User_name"]
    saved_STT_model_name = profile_dict["STT_model_name"]
    saved_realtime_STT_model_name = profile_dict["Realtime_STT_model_name"]

    #set the user choice before clearing the data to profile data

    set_character_name = settingform.character_name_input.data
    set_thinking_model_name = settingform.thinking_model_name_input.data
    set_vision_model_name = settingform.vision_model_name_input.data
    set_character_desc = settingform.character_desc_input.data
    set_TTS_option = settingform.TTS_onoff.data
    set_TTS_model_name = settingform.TTS_model_name_input.data
    set_user_name = settingform.user_name_input.data
    set_STT_model_name = settingform.STT_model_name_input.data
    set_realtime_STT_model_name = settingform.realtime_STT_model_name_input.data

    #set the default of these input places for user loading the setting pages

    settingform.character_desc_input.data = saved_character_desc
    settingform.character_name_input.data = saved_character_name
    settingform.thinking_model_name_input.data = saved_thinking_model_name
    settingform.vision_model_name_input.data = saved_vision_model_name
    settingform.TTS_onoff.data = saved_tts_option
    settingform.TTS_model_name_input.data = saved_tts_model_name
    settingform.user_name_input.data = saved_user_name
    settingform.STT_model_name_input.data = saved_STT_model_name
    settingform.realtime_STT_model_name_input.data = saved_realtime_STT_model_name

    if settingform.validate_on_submit():

        if settingform.ClearBtn.data:

            #clear the chat log
            with open("./log.json", "w", encoding="utf-8") as f:
                json.dump({}, f)
            

        elif settingform.Clear_TTS_wavs_Btn.data:

            for filename in os.listdir(app.config["VOICEMODELSOUTPUT_FOLDER"]):

                file_path = os.path.join(app.config["VOICEMODELSOUTPUT_FOLDER"], filename)

                try:

                    os.unlink(file_path)

                except OSError:
                    output_message = "ErrorRM: Failed to remove all files in voice models output folder"
                    return render_template('settings.html', settingform=settingform,output=output_message)

        elif settingform.Clear_STT_wavs_Btn.data:

             for filename in os.listdir(app.config["VOICEINPUT_FOLDER"]):

                file_path = os.path.join(app.config["VOICEINPUT_FOLDER"], filename)

                try:

                    os.unlink(file_path)

                except OSError:
                    output_message = "ErrorRM: Failed to remove all files in input folder"
                    return render_template('settings.html', settingform=settingform,output=output_message)

        elif settingform.ClearFilesBtn.data:

            for filename in os.listdir(app.config["UPLOAD_FOLDER"]):

                file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

                try:

                    os.unlink(file_path)

                except OSError:
                    output_message = "ErorrRM: Failed to remove all files in upload folder"
                    return render_template('settings.html', settingform=settingform,output=output_message)

        elif settingform.Submit.data:

            #Check if settings are valid

            if set_TTS_model_name not in os.listdir(app.config["CUSTOMVOICEMODELS_FOLDER"]):

                print("TTS MODEL NOT FOUND!")

                output_message = f"TTS model not found in custom voice models file.Please check the file."

                set_TTS_model_name = "en_US-sevenseven-medium.onnx"

            prev_profile_dict = {}

            with open("./profile.json", "r", encoding="utf-8") as f:
                try:
                    #load the profile file back to a dict
                    prev_profile_dict = json.load(f)
                except json.JSONDecodeError:
                    prev_profile_dict = {}
                    output_message = "ErrorDECODE: profile.json is empty or corrupted"
                    return render_template('settings.html', settingform=settingform,output=output_message)


            prev_profile_dict["character_name"] = set_character_name
            prev_profile_dict["thinking_model_name"] = set_thinking_model_name
            prev_profile_dict["character_desc"] = set_character_desc
            prev_profile_dict["TTS"] = set_TTS_option
            prev_profile_dict["TTS_model_name"] = set_TTS_model_name
            prev_profile_dict["User_name"] = set_user_name
            prev_profile_dict["STT_model_name"] = set_STT_model_name
            prev_profile_dict["vision_model_name"] = set_vision_model_name
            prev_profile_dict["Realtime_STT_model_name"] = set_realtime_STT_model_name

            if saved_realtime_STT_model_name != set_realtime_STT_model_name:

                del realtime_STT_recorder

                gc.collect()

                realtime_STT_recorder = AudioToTextRecorder(model=set_realtime_STT_model_name,device=device,compute_type="float16" if torch.cuda.is_available() else "float32")

            with open("./profile.json", "w", encoding="utf-8") as f:
                json.dump(prev_profile_dict, f)
            
            #update the form

            settingform.character_desc_input.data = set_character_desc
            settingform.character_name_input.data = set_character_name
            settingform.thinking_model_name_input.data = set_thinking_model_name
            settingform.TTS_onoff.data = set_TTS_option
            settingform.TTS_model_name_input.data = set_TTS_model_name
            settingform.user_name_input.data = set_user_name
            settingform.STT_model_name_input.data = set_STT_model_name
            settingform.vision_model_name_input.data = set_vision_model_name
            settingform.realtime_STT_model_name_input.data = set_realtime_STT_model_name

            check_if_model_downloaded()
           

        else:
            output_message = "Error404: Unexpected error."
            return render_template('settings.html', settingform=settingform,output=output_message)

    return render_template('settings.html', settingform=settingform,output=output_message)

if __name__ == "__main__":

    app.run(host=RUN_HOST,port=RUN_PORT,debug=True)
    