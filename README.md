# PersonalizedChatBot
A chatbot with RAG system web app!

# Updates V3
1. Fixed some webUI glitches
2. New Realtime-STT feature(AI Assistant)
3. Added the ability for chatbot to look at your screen
4. Added the ability for chatbot to open and read .pdf files
5. New action analyse
6. New short-term and long-term memory structure
7. New character viewer(Pngtuber)(BETA)
8. Minimized require ROM and RAM memory
9. Imported RAG system into it.
10. New personality analysis

# Introduction

Hello! I am Anson.In the past few weeks,I build a small web app for people who want there LLM(chatbot) to be locally host and customizable!
This web app implented its own memory(long-short term memory),personality and emotions system.
I have try to make the hardware requirement for this web app to be as low as possible.But if you want a better experiment,it is recommended to use a good GPU(cuda) for this program.
This project is better to use a database like chroma but I think JSON files is better for representation.If you want,you can build your own database for this!

Have fun!

# Requirement

1. Python3 (other packages should be auto installed.If the auto installation doesn't work, please install the packages manually)

## Other packages

Check the [requirement.txt](linkhere) for all the requirements
Download all the requirements packages by 
```
pip install -r requirements.txt
```
# Start-up

1. Download this git as .zip file and extract it
2. Delete the deleteme.txt in input,output,UploadFiles folder
3. Run the webServer.py
4. Go to website http://127.0.0.1:5000/

# Usage

## Start chatting
Chat with the bot in the index page or homepage or click the chat text on the navbar

## Settings
Click the setting text on the navbar or go to settings.html to change the settings of the chatbot

# Notes
If the new webServer.py doesn't work,you can use the webServer_old.py in the "Archives" Folder.

# Special thanks
Piper-tts,faster-whisper,ollama,RealtimeSTT

# License

[MIT](https://github.com/sunshinghin/PersonalizedChatBot/blob/main/LICENSE)

#Thanks you
Thanks you for using this service! 🥰💖
