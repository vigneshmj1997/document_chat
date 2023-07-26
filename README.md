# document_chat
An document chatting application using langchain

# To run the code
install conda using the (this)[https://docs.conda.io/en/latest/miniconda.html]

create a conda env

`conda create -n chat python=3.8`

activate conda env

`conda activate chat`

install all python packages

`pip install -r requirements.txt`

# There are two ways to run the code 

To run it as a chat bot

```
cd src
python3 chat_doc.py --path <path of the knowledge base>
```

To run test the output for the given langguage model

```
cd src
python3 chat_doc.py --path <path of the knowledge base> --test_path <path of the csv>
```

for more help 

```
python3 chat_doc.py --help
```

