# SERPデータを使用してGPT-4oでRAGチャットボットを作成する

[![Promo](https://media.brightdata.com/2025/08/SERP-API-50-off-GitHub-banner_1389_166.png)](https://brightdata.jp/) 

このガイドでは、より正確でコンテキストが豊富なAIレスポンスを得るために、GPT-4oとBright DataのSERP APIを使用してPythonのRAGチャットボットを構築する方法を説明します。

1. [Introduction](#how-to-creating-a-rag-chatbot-with-gpt-4o-using-serp-data)
2. [What Is RAG?](#what-is-rag)
3. [Why Feed AI Models With SERP Data](#why-feed-ai-models-with-serp-data)
4. [RAG With SERP Data With GPT Models Using Python: Step-By-Step Tutorial](#rag-with-serp-data-with-gpt-models-using-python-step-by-step-tutorial)
    1. [Step #1: Initialize a Python Project](#step-1-initialize-a-python-project)
    2. [Step #2: Install the Required Libraries](#step-2-install-the-required-libraries)
    3. [Step #3: Prepare Your Project](#step-3-prepare-your-project)
    4. [Step #4: Configure SERP API](#step-4-configure-serp-api)
    5. [Step #5: Implement the SERP Scraping Logic](#step-5-implement-the-serp-scraping-logic)
    6. [Step #6: Extract Text from the SERP URLs](#step-6-extract-text-from-the-serp-urls)
    7. [Step #7: Generate the RAG Prompt](#step-7-generate-the-rag-prompt)
    8. [Step #8: Perform the GPT Request](#step-8-perform-the-gpt-request)
    9. [Step #9: Create the Application UI](#step-9-create-the-application-ui)
    10. [Step #10: Put It All Together](#step-10-put-it-all-together)
    11. [Step #11: Test the Application](#step-11-test-the-application)
5. [Conclusion](#conclusion)

## What Is RAG?

RAGは、[Retrieval-Augmented Generation](https://blogs.nvidia.comhttps://brightdata.jp/blog/what-is-retrieval-augmented-generation/)（検索拡張生成）の略で、情報検索とテキスト生成を組み合わせたAIアプローチです。RAGワークフローでは、アプリケーションがまず、ドキュメント、Webページ、データベースなどの外部ソースから関連データを取得します。その後、そのデータをAIモデルに渡し、よりコンテキストに即したレスポンスを生成できるようにします。

RAGは、GPTのような大規模言語モデル（LLM）が、元の学習データを超えて最新情報へアクセスして参照できるようにすることで強化します。このアプローチは、正確でコンテキスト特化の情報が必要なシナリオで重要であり、AI生成レスポンスの品質と精度の両方を改善します。

## Why Feed AI Models With SERP Data

GPT-4oのknowledge cutoff dateは[October 2023](https://computercity.com/artificial-intelligence/knowledge-cutoff-dates-llms)であり、その時点以降に出た出来事や情報にはアクセスできないことを意味します。しかし、[GPT-4o models](https://openai.com/index/hello-gpt-4o/)はBing検索連携を使用してインターネットからリアルタイムにデータを取り込むことができます。これにより、より最新の情報を提供でき、詳細で正確、かつコンテキストが豊富なレスポンスを提示しやすくなります。

## RAG With SERP Data With GPT Models Using Python: Step-By-Step Tutorial

このチュートリアルでは、OpenAIのGPTモデルを使用してRAGチャットボットを構築する手順を案内します。アイデアは、特定の検索クエリに対してGoogleで上位表示されるページからテキストを収集し、それをGPTリクエストのコンテキストとして使用することです。

最大の課題はSERPデータのスクレイピングです。多くの検索エンジンには、自動アクセスを防ぐための高度なアンチボットソリューションが組み込まれています。詳細なガイダンスについては、[how to scrape Google in Python](https://brightdata.jp/blog/web-data/scraping-google-with-python)のガイドをご参照ください。

スクレイピングプロセスを簡素化するために、[Bright Data’s SERP API](https://brightdata.jp/products/serp-api)を使用します。

このSERPスクレイパーを使用すると、シンプルなHTTPリクエストでGoogle、DuckDuckGo、Bing、Yandex、Baidu、その他の検索エンジンからSERPを簡単に取得できます。

続いて、返されたURLから[headless browser](https://brightdata.jp/blog/web-data/best-headless-browsers)を使用してテキストデータを抽出します。その後、その情報をRAGワークフローにおけるGPTモデルのコンテキストとして使用します。代わりにAIを使用してオンラインデータを直接取得したい場合は、[web scraping with ChatGPT](https://brightdata.jp/blog/web-data/web-scraping-with-chatgpt)に関する記事をご覧ください。

このガイド内のコードはすべて、GitHubリポジトリでも入手できます：

```bash
git clone https://github.com/Tonel/rag_gpt_serp_scraping
```

README.mdファイルの手順に従って、プロジェクトの依存関係をインストールし、プロジェクトを起動してください。

なお、このブログ記事で紹介しているアプローチは、他の検索エンジンやLLMにも簡単に適用できます。

> **Note**:\
> このガイドはUnixおよびmacOSを前提としています。Windowsユーザーの方も、[Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install)を使用することでチュートリアルを進められます。

### Step #1: Initialize a Python Project

マシンにPython 3がインストールされていることを確認してください。インストールされていない場合は、[download and install it](https://www.python.org/downloads/)してください。

プロジェクト用のフォルダーを作成し、ターミナルでそのフォルダーに切り替えます：

```bash
mkdir rag_gpt_serp_scraping

cd rag_gpt_serp_scraping
```

`rag_gpt_serp_scraping`フォルダーにはPythonのRAGプロジェクトが含まれます。

次に、お好みのPython IDEでプロジェクトディレクトリを読み込みます。[PyCharm Community Edition](https://www.jetbrains.com/pycharm/download/)または[Visual Studio Code with the Python extension](https://code.visualstudio.com/docs/languages/python)で問題ありません。

rag\_gpt\_serp\_scraping内に空のapp.pyファイルを追加します。このファイルにスクレイピングとRAGのロジックを記述します。

続いて、プロジェクトディレクトリで[Python virtual environment](https://docs.python.org/3/library/venv.html)を初期化します：

```bash
python3 -m venv env
```

以下のコマンドで仮想環境を有効化します：

```bash
source ./env/bin/activate
```

### Step #2: Install the Required Libraries

このPython RAGプロジェクトでは、以下の依存関係を使用します：

*   [`python-dotenv`](https://pypi.org/project/python-dotenv/): Bright Dataの認証情報やOpenAI API keyなどの機密情報を安全に管理するために使用します。
*   [`requests`](https://pypi.org/project/requests/): Bright DataのSERP APIに対してHTTPリクエストを実行するために使用します。
*   [`langchain-community`](https://pypi.org/project/langchain-community/): GoogleのSERPページからテキストを取得し、RAG向けに関連コンテンツを生成できるようクリーニングするために使用します。
*   [`openai`](https://pypi.org/project/openai/): 与えられた入力とRAGコンテキストに基づいて自然言語レスポンスを生成するため、GPTモデルと連携する目的で使用します。
*   [`streamlit`](https://pypi.org/project/streamlit/): ユーザーがGoogle検索クエリとAIプロンプトを入力し、結果を動的に表示できるUIを作成する際に便利です。

すべての依存関係をインストールします：

```bash
pip install python-dotenv requests langchain-community openai streamlit
```

langchain-communityの[AsyncChromiumLoader](https://python.langchain.com/docs/integrations/document_loaders/async_chromium/)を使用しますが、これには以下の依存関係が必要です：

```bash
pip install --upgrade --quiet playwright beautifulsoup4 html2text
```

また、Playwrightが正しく動作するにはブラウザのインストールも必要です：

```bash
playwright install
```

### Step #3: Prepare Your Project

`app.py`に以下のimportを追加します：

```python
from dotenv import load_dotenv

import os

import requests

from langchain_community.document_loaders import AsyncChromiumLoader

from langchain_community.document_transformers import BeautifulSoupTransformer

from openai import OpenAI

import streamlit as st
```

次に、すべての認証情報を保存するために、プロジェクトフォルダーに`.env`ファイルを作成します。プロジェクト構成は以下のようになります：

![Project structure](https://github.com/luminati-io/rag-chatbot/blob/main/Images/image-19.png)

`app.py`で以下の関数を使用し、`python-dotenv`に`.env`から環境変数を読み込むよう指示します：

```python
load_dotenv()
```

これで、`.env`またはシステムから環境変数を次のようにインポートできます：

```python
os.environ.get("<ENV_NAME>")
```

### Step #4: Configure SERP API

Bright Data’s SERP APIを使用して検索エンジンの検索結果ページからコンテンツを取得し、それをPythonのRAGワークフローで利用します。具体的には、SERP APIが返すWebページURLからテキストを抽出します。

SERP APIのセットアップについては、[official documentation](https://docs.brightdata.com/scraping-automation/serp-api/quickstart)をご参照ください。あるいは、以下の手順に従ってください。

まだアカウントを作成していない場合は、[sign up for Bright Data](https://brightdata.jp)してください。ログイン後、アカウントのダッシュボードに移動します：

![Account main dashboard](https://github.com/luminati-io/rag-chatbot/blob/main/Images/image-18.png)

そこで「Get proxy products」ボタンをクリックします。

すると以下のページに移動するので、「SERP API」行をクリックします：

![Clicking on SERP API](https://github.com/luminati-io/rag-chatbot/blob/main/Images/image-17.png)

SERP APIのプロダクトページで、「Activate zone」を切り替えてプロダクトを有効化します：

![Activating the SERP zone](https://github.com/luminati-io/rag-chatbot/blob/main/Images/image-16.png)

次に、「Access parameters」セクションにあるSERP APIのhost、port、username、passwordをコピーし、`.env`ファイルに追加します：

```python
BRIGHT_DATA_SERP_API_HOST="<YOUR_HOST>"

BRIGHT_DATA_SERP_API_PORT=<YOUR_PORT>

BRIGHT_DATA_SERP_API_USERNAME="<YOUR_USERNAME>"

BRIGHT_DATA_SERP_API_PASSWORD="<YOUR_PASSWORD>"
```

`<YOUR_XXXX>`プレースホルダーを、SERP APIページでBright Dataが提供する値に置き換えてください。

「Access parameters」のhostは、次のような形式である点にご注意ください：

```python
brd.superproxy.io:33335
```

以下のように分割してください：

```python
BRIGHT_DATA_SERP_API_HOST="brd.superproxy.io"

BRIGHT_DATA_SERP_API_PORT=33335
```

### Step #5: Implement the SERP Scraping Logic

`app.py`に以下の関数を追加して、GoogleのSERPページから最初の`number_of_urls`件のURLを取得します：

```python
def get_google_serp_urls(query, number_of_urls=5):

# perform a Bright Data's SERP API request

# with JSON autoparsing

host = os.environ.get("BRIGHT_DATA_SERP_API_HOST")

port = os.environ.get("BRIGHT_DATA_SERP_API_PORT")

username = os.environ.get("BRIGHT_DATA_SERP_API_USERNAME")

password = os.environ.get("BRIGHT_DATA_SERP_API_PASSWORD")

proxy_url = f"http://{username}:{password}@{host}:{port}"

proxies = {"http": proxy_url, "https": proxy_url}

url = f"https://www.google.com/search?q={query}&brd_json=1"

response = requests.get(url, proxies=proxies, verify=False)

# retrieve the parsed JSON response

response_data = response.json()

# extract a "number_of_urls" number of

# Google SERP URLs from the response

google_serp_urls = []

if "organic" in response_data:

for item in response_data["organic"]:

if "link" in item:

google_serp_urls.append(item["link"])

return google_serp_urls[:number_of_urls]
```

この関数は、query引数で指定された検索クエリを使ってSERP APIにHTTP GETリクエストを送信します。[`brd_json=1`](https://docs.brightdata.com/scraping-automation/serp-api/parsing-search-results)クエリパラメータにより、SERP APIが結果を以下の形式でJSONにパースしてくれます：

```json
{

"general": {

"search_engine": "google",

"results_cnt": 1980000000,

"search_time": 0.57,

"language": "en",

"mobile": false,

"basic_view": false,

"search_type": "text",

"page_title": "pizza - Google Search",

"code_version": "1.90",

"timestamp": "2023-06-30T08:58:41.786Z"

},

"input": {

"original_url": "https://www.google.com/search?q=pizza&brd_json=1",

"user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12) AppleWebKit/608.2.11 (KHTML, like Gecko) Version/13.0.3 Safari/608.2.11",

"request_id": "hl_1a1be908_i00lwqqxt1"

},

"organic": [

{

"link": "https://www.pizzahut.com/",

"display_link": "https://www.pizzahut.com",

"title": "Pizza Hut | Delivery & Carryout - No One OutPizzas The Hut!",

"image": "omitted for brevity...",

"image_alt": "pizza from www.pizzahut.com",

"image_base64": "omitted for brevity...",

"rank": 1,

"global_rank": 1

},

{

"link": "https://www.dominos.com/en/",

"display_link": "https://www.dominos.com › ...",

"title": "Domino's: Pizza Delivery & Carryout, Pasta, Chicken & More",

"description": "Order pizza, pasta, sandwiches & more online for carryout or delivery from Domino's. View menu, find locations, track orders. Sign up for Domino's email ...",

"image": "omitted for brevity...",

"image_alt": "pizza from www.dominos.com",

"image_base64": "omitted for brevity...",

"rank": 2,

"global_rank": 3

},

// omitted for brevity...

],

// omitted for brevity...

}
```

関数の最後の数行では、結果のJSONデータから各SERP URLを取得し、最初の`number_of_urls`件のURLだけを選択して、リストとして返します。

### Step #6: Extract Text from the SERP URLs

各SERP URLからテキストを抽出する関数を定義します：

```python
# Note: Some websites may have dynamic content or anti-scraping measures that could prevent text extraction.
# In such cases, please consider using additional tools like Selenium
def extract_text_from_urls(urls, number_of_words=600): 

# instruct a headless Chrome instance to visit the provided URLs

# with the specified user-agent

loader = AsyncChromiumLoader(

urls,

user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",

)

html_documents = loader.load()

# process the extracted HTML documents to extract text from them

bs_transformer = BeautifulSoupTransformer()

docs_transformed = bs_transformer.transform_documents(

html_documents,

tags_to_extract=["p", "em", "li", "strong", "h1", "h2"],

unwanted_tags=["a"],

remove_comments=True,

)

# make sure each HTML text document contains only a number

# number_of_words words

extracted_text_list = []

for doc_transformed in docs_transformed:

# split the text into words and join the first number_of_words

words = doc_transformed.page_content.split()[:number_of_words]

extracted_text = " ".join(words)

# ignore empty text documents

if len(extracted_text) != 0:

extracted_text_list.append(extracted_text)

return extracted_text_list
```

この関数は次の処理を行います：

1.  引数として渡されたURLから、headless Chromeブラウザインスタンスを使ってWebページを読み込みます。
2.  [BeautifulSoupTransformer](https://python.langchain.com/v0.2/api_reference/community/document_transformers/langchain_community.document_transformers.beautiful_soup_transformer.BeautifulSoupTransformer.html)を利用して各ページのHTMLを処理し、特定のタグ（`<p>`, `<h1>`, `<strong>`など）からテキストを抽出し、不要なタグ（`<a>`など）やコメントを除外します。
3.  各Webページから抽出するテキストを、`number_of_words`引数で指定した単語数に制限します。
4.  各URLから抽出したテキストのリストを返します。

`["p", "em", "li", "strong", "h1", "h2"]`タグでほとんどのWebページからテキストを抽出できますが、特定のシナリオではこのHTMLタグのリストをカスタマイズする必要がある場合があります。また、各テキスト項目の対象単語数を増減する必要があるかもしれません。

たとえば、以下の[Webページ](https://athomeinhollywood.com/2024/09/19/transformers-one-review/)を考えてみてください：

![Transformers one review page](https://github.com/luminati-io/rag-chatbot/blob/main/Images/image-15.png)

この関数をそのページに適用すると、次のテキスト配列になります：

```python
["Lisa Johnson Mandell’s Transformers One review reveals the heretofore inconceivable: It’s one of the best animated films of the year! I never thought I’d see myself write this about a Transformers movie, but Transformers One is actually an exceptional film! ..."]
```

`extract_text_from_urls()`が返すテキスト項目のリストは、OpenAIモデルに投入するRAGコンテキストを表します。

### Step #7: Generate the RAG Prompt

AIプロンプトのリクエストとテキストコンテキストを、最終的なRAGプロンプトに変換する関数を定義します：

```python
def get_openai_prompt(request, text_context=[]):

# default prompt

prompt = request

# add the context to the prompt, if present

if len(text_context) != 0:

context_string = "\n\n--------\n\n".join(text_context)

prompt = f"Answer the request using only the context below.\n\nContext:\n{context_string}\n\nRequest: {request}"

return prompt
```

RAGコンテキストが指定された場合、前の関数が返すプロンプトは次の形式になります：

```
Answer the request using only the context below.

Context:

Bla bla bla...

--------

Bla bla bla...

--------

Bla bla bla...

Request: <YOUR_REQUEST>
```

### Step #8: Perform the GPT Request

まず、`app.py`ファイルの先頭でOpenAIクライアントを初期化します：

```python
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
```

これは`OPENAI_API_KEY`環境変数に依存しており、システムの環境変数として直接定義するか、`.env`ファイルに定義できます：

`OPENAI_API_KEY="<YOUR_API_KEY>"`

`<YOUR_API_KEY>`を、あなたの[OpenAI API key](https://platform.openai.com/api-keys)の値に置き換えてください。取得方法が分からない場合は、[official guide](https://platform.openai.com/docs/quickstart)に従ってください。

次に、OpenAI公式クライアントを使用して、[GPT-4o mini](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/) AIモデルへリクエストを実行する関数を書きます：

```python
def interrogate_openai(prompt, max_tokens=800):

# interrogate the OpenAI model with the given prompt

response = openai_client.chat.completions.create(

model="gpt-4o-mini",

messages=[{"role": "user", "content": prompt}],

max_tokens=max_tokens,

)

return response.choices[0].message.content
```

> **Note**:\
> OpenAI APIがサポートする他のGPTモデルも設定できます。

`get_openai_prompt()`が返す、指定されたテキストコンテキストを含むプロンプトで呼び出した場合、`interrogate_openai()`は意図どおりに検索拡張生成を正常に実行します。

### Step #9: Create the Application UI

Streamlitを使用して、ユーザーが以下を指定できるシンプルな[form UI](https://docs.streamlit.io/develop/concepts/architecture/forms)を定義します：

1.  SERP APIに渡すGoogle検索クエリ
2.  GPT-4o miniへ送信するAIプロンプト

そのために、次のコードを使用します：

```python
with st.form("prompt_form"):

# initialize the output results

result = ""

final_prompt = ""

# textarea for user to input their Google search query

google_search_query = st.text_area("Google Search:", None)

# textarea for user to input their AI prompt

request = st.text_area("AI Prompt:", None)

# button to submit the form

submitted = st.form_submit_button("Send")

# if the form is submitted

if submitted:

# retrieve the Google SERP URLs from the given search query

google_serp_urls = get_google_serp_urls(google_search_query)

# extract the text from the respective HTML pages

extracted_text_list = extract_text_from_urls(google_serp_urls)

# generate the AI prompt using the extracted text as context

final_prompt = get_openai_prompt(request, extracted_text_list)

# interrogate an OpenAI model with the generated prompt

result = interrogate_openai(final_prompt)

# dropdown containing the generated prompt

final_prompt_expander = st.expander("AI Final Prompt:")

final_prompt_expander.write(final_prompt)

# write the result from the OpenAI model

st.write(result)
```

これでPythonのRAGスクリプトは準備完了です。

### Step #10: Put It All Together

`app.py`ファイルには以下のコードが含まれているはずです：

```python
from dotenv import load_dotenv

import os

import requests

from langchain_community.document_loaders import AsyncChromiumLoader

from langchain_community.document_transformers import BeautifulSoupTransformer

from openai import OpenAI

import streamlit as st

# load the environment variables from the .env file

load_dotenv()

# initialize the OpenAI API client with your API key

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_google_serp_urls(query, number_of_urls=5):

# perform a Bright Data's SERP API request

# with JSON autoparsing

host = os.environ.get("BRIGHT_DATA_SERP_API_HOST")

port = os.environ.get("BRIGHT_DATA_SERP_API_PORT")

username = os.environ.get("BRIGHT_DATA_SERP_API_USERNAME")

password = os.environ.get("BRIGHT_DATA_SERP_API_PASSWORD")

proxy_url = f"http://{username}:{password}@{host}:{port}"

proxies = {"http": proxy_url, "https": proxy_url}

url = f"https://www.google.com/search?q={query}&brd_json=1"

response = requests.get(url, proxies=proxies, verify=False)

# retrieve the parsed JSON response

response_data = response.json()

# extract a "number_of_urls" number of

# Google SERP URLs from the response

google_serp_urls = []

if "organic" in response_data:

for item in response_data["organic"]:

if "link" in item:

google_serp_urls.append(item["link"])

return google_serp_urls[:number_of_urls]

def extract_text_from_urls(urls, number_of_words=600):

# instruct a headless Chrome instance to visit the provided URLs

# with the specified user-agent

loader = AsyncChromiumLoader(

urls,

user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",

)

html_documents = loader.load()

# process the extracted HTML documents to extract text from them

bs_transformer = BeautifulSoupTransformer()

docs_transformed = bs_transformer.transform_documents(

html_documents,

tags_to_extract=["p", "em", "li", "strong", "h1", "h2"],

unwanted_tags=["a"],

remove_comments=True,

)

# make sure each HTML text document contains only a number

# number_of_words words

extracted_text_list = []

for doc_transformed in docs_transformed:

# split the text into words and join the first number_of_words

words = doc_transformed.page_content.split()[:number_of_words]

extracted_text = " ".join(words)

# ignore empty text documents

if len(extracted_text) != 0:

extracted_text_list.append(extracted_text)

return extracted_text_list

def get_openai_prompt(request, text_context=[]):

# default prompt

prompt = request

# add the context to the prompt, if present

if len(text_context) != 0:

context_string = "\n\n--------\n\n".join(text_context)

prompt = f"Answer the request using only the context below.\n\nContext:\n{context_string}\n\nRequest: {request}"

return prompt

def interrogate_openai(prompt, max_tokens=800):

# interrogate the OpenAI model with the given prompt

response = openai_client.chat.completions.create(

model="gpt-4o-mini",

messages=[{"role": "user", "content": prompt}],

max_tokens=max_tokens,

)

return response.choices[0].message.content

# create a form in the Streamlit app for user input

with st.form("prompt_form"):

# initialize the output results

result = ""

final_prompt = ""

# textarea for user to input their Google search query

google_search_query = st.text_area("Google Search:", None)

# textarea for user to input their AI prompt

request = st.text_area("AI Prompt:", None)

# button to submit the form

submitted = st.form_submit_button("Send")

# if the form is submitted

if submitted:

# retrieve the Google SERP URLs from the given search query

google_serp_urls = get_google_serp_urls(google_search_query)

# extract the text from the respective HTML pages

extracted_text_list = extract_text_from_urls(google_serp_urls)

# generate the AI prompt using the extracted text as context

final_prompt = get_openai_prompt(request, extracted_text_list)

# interrogate an OpenAI model with the generated prompt

result = interrogate_openai(final_prompt)

# dropdown containing the generated prompt

final_prompt_expander = st.expander("AI Final Prompt")

final_prompt_expander.write(final_prompt)

# write the result from the OpenAI model

st.write(result)
```

### Step #11: Test the Application

以下でPython RAGアプリケーションを起動します：

```bash
# Note: Streamlit is designed for lightweight applications. For production-grade deployments, consider using frameworks like Flask or FastAPI.
streamlit run app.py
```
ターミナルには次の出力が表示されるはずです：

```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501

Network URL: http://172.27.134.248:8501
```

指示に従い、ブラウザで`http://localhost:8501`にアクセスしてください。以下のような画面が表示されるはずです：

![Streamlit app screenshot](https://github.com/luminati-io/rag-chatbot/blob/main/Images/image-14.png)

次のようなGoogle検索クエリを使用してアプリケーションをテストします：

```
Transformers One review
```

そして、AIプロンプトは次のとおりです：

```
Write a review for the movie Transformers One
```

「Send」をクリックし、アプリケーションがリクエストを処理するのを待ちます。数秒後、以下のような結果が得られるはずです：

![App result screenshot](https://github.com/luminati-io/rag-chatbot/blob/main/Images/image-13.png)

「AI Final Prompt」ドロップダウンを展開すると、アプリケーションがRAGに使用した完全なプロンプトを確認できます。

## Conclusion

PythonのRAGチャットボットを使用する際の主要な課題は、Googleのような検索エンジンをスクレイピングすることです：

1. SERPページの構造が頻繁に変更されます。
2. 利用可能な中でも最も洗練されたアンチボット対策の一部によって保護されています。
3. 大量のSERPデータを同時接続で取得するのは複雑で、コストが高くなる場合があります。

[Bright Data’s SERP API](https://brightdata.jp/products/serp-api)は、主要な検索エンジンすべてからリアルタイムのSERPデータを手間なく取得するのに役立ちます。また、RAGやその他多くのアプリケーションもサポートします。今すぐ無料トライアルを開始してください！