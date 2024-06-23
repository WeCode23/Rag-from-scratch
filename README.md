# Rag-from-scratch
A simple Rag tutorial for beginners with the knowledge gained through Course 1 of Coursera's LLMops series by Duke University. 


# Retrieval Augmented Generation (RAG) with qDrant vector db and llamafile model

- Here we are using qDrant, a vector DB, to generate the 
    - Context, which acts as a supporting context for the user query.

- We will run a LLM model locally using Llamafile, here is a link to start a model easily

   https://github.com/Mozilla-Ocho/llamafile


## Install the prerequisites

Use the `requirements.txt` to install all dependencies

```bash
python -m venv .venv
./.venv/bin/pip install -r requirements.txt
```


Steps to run a server to receive user input and generate relevant response.

1. We will be building context based on the data of Wines which I have taken from Coursera LLMops course.
   - a csv file is placed in the repository.

2. Open a terminal and run the following command to start the LLM model server. assuming you have download 'TinyLlama-1.1B-Chat-v1.0.F16.llamafile'
   
   $ chmod +x TinyLlama-1.1B-Chat-v1.0.F16.llamafile
   $ ./TinyLlama-1.1B-Chat-v1.0.F16.llamafile

    This will run the llm model server we will send the our input to this model via our web app server.

3. You will find a main.py which is a fastapi server script. you will find the relevant code inside it.
    Run following command on new terminal.
    
    $ uvicorn main:app --host 0.0.0.0 --port 8000


4. The server will be up at localhost:8000

    you can pass your query inside the "query" key and execute it.

    an example of query is "Suggest a good shiraz wine"

